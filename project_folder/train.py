import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def train_model(
    num_epochs, 
    batch_size, 
    model, 
    optimizer, 
    criterion, 
    train_data, 
    val_data, 
    scheduler=None,
    scheduler_step_on_iter=False,
    predicts_clean=False,
    measure_latency=False,
    time_limit=None,
    seed=42, 
    device='cpu'
):
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, generator=g)

    print(f"Using device: {device}")
    model.to(device)

    avg_latency = 0.0
    
    if measure_latency:
        print("\n--- Measuring Latency ---")
        model.eval()
        
        latency_batch_size = 1
        measure_bs = 16 
        latency_loader = DataLoader(val_data, batch_size=measure_bs, shuffle=False)
        latency_iter = iter(latency_loader)
        
        num_warmup = 10
        num_test = 20
        
        print(f"Running {num_warmup} warm-up batches...")
        with torch.no_grad():
            for _ in range(num_warmup):
                try:
                    deg, _ = next(latency_iter)
                except StopIteration:
                    latency_iter = iter(latency_loader)
                    deg, _ = next(latency_iter)
                
                deg = deg.to(device)
                model(deg)
        
        print(f"Running {num_test} test batches...")
        total_time = 0.0
        
        with torch.no_grad():
            for i in range(num_test):
                try:
                    deg, _ = next(latency_iter)
                except StopIteration:
                    latency_iter = iter(latency_loader)
                    deg, _ = next(latency_iter)
                
                deg = deg.to(device)
                
                if device == 'cuda': torch.cuda.synchronize()
                elif device == 'mps': torch.mps.synchronize()
                
                start_t = time.perf_counter()
                model(deg)
                
                if device == 'cuda': torch.cuda.synchronize()
                elif device == 'mps': torch.mps.synchronize()
                
                end_t = time.perf_counter()
                total_time += (end_t - start_t)
        
        avg_latency = total_time / num_test
        print(f"Average latency ({measure_bs} imgs): {avg_latency:.6f} sec")

        if time_limit is not None and avg_latency > time_limit:
            print(f"Latency limit exceeded ({avg_latency:.4f} > {time_limit}). Aborting.")
            return [(100.0, 100.0, 100.0, avg_latency)]

    print("\n--- Starting Training ---")
    history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for degraded, clean in train_loop:
            degraded, clean = degraded.to(device), clean.to(device)
            
            optimizer.zero_grad()
            
            prediction = model(degraded)
            
            if predicts_clean:
                restored = prediction
            else:
                restored = degraded + prediction
            
            loss_dict = criterion(restored, clean)
            loss = loss_dict['total']
            
            loss.backward()
            optimizer.step()
            
            if scheduler is not None and scheduler_step_on_iter:
                scheduler.step()
            
            running_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = running_train_loss / len(train_loader)
        
        if scheduler is not None and not scheduler_step_on_iter:
            scheduler.step()

        model.eval()
        val_totals = {'loss': 0.0, 'pixel': 0.0, 'perception': 0.0}
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            for degraded, clean in val_loop:
                degraded, clean = degraded.to(device), clean.to(device)
                
                prediction = model(degraded)
                if predicts_clean:
                    restored = prediction
                else:
                    restored = degraded + prediction
                
                loss_dict = criterion(restored, clean)
                
                val_totals['loss'] += loss_dict['total'].item()
                val_totals['pixel'] += loss_dict.get('pixel', 0.0) 
                val_totals['perception'] += loss_dict.get('perception', 0.0) 
                
                val_loop.set_postfix(
                    loss=loss_dict['total'].item(),
                    perc=loss_dict.get('perception', 0.0)
                )

        avg_val_loss = val_totals['loss'] / len(val_loader)
        avg_val_pixel = val_totals['pixel'] / len(val_loader)
        avg_val_perception = val_totals['perception'] / len(val_loader)
        
        if measure_latency:
            epoch_record = (
                avg_train_loss, 
                avg_val_loss, 
                avg_val_perception, 
                avg_latency
            )
            print(
                f"Epoch {epoch+1} -> "
                f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, "
                f"Perc: {avg_val_perception:.4f}, Lat: {avg_latency:.4f}"
            )
        else:
            epoch_record = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_pixel': avg_val_pixel,
                'val_perception': avg_val_perception
            }
            print(
                f"Epoch {epoch+1} -> "
                f"Train: {avg_train_loss:.5f}, Val: {avg_val_loss:.5f}, "
                f"Pixel: {avg_val_pixel:.5f}, Perc: {avg_val_perception:.5f}"
            )
            
        history.append(epoch_record)

    return history