"""Training and validation loops."""
import os
import csv
import torch
from src.metrics.xview2 import XView2Metric

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for pre, post, mask in loader:
        pre = pre.to(device)
        post = post.to(device)
        mask = mask.to(device)

        logits = model(pre, post)
        loss = loss_fn(logits, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    metric = XView2Metric()

    with torch.no_grad():
        for pre, post, mask in loader:
            pre = pre.to(device)
            post = post.to(device)
            mask = mask.to(device)

            logits = model(pre, post)
            pred = logits.argmax(dim=1)
            metric.update(pred, mask)

    return metric.compute_f1()

# log 
def append_log(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# early stopping (patience = 20%)
def fit(model, train_loader, val_loader, loss_fn, optimizer, device, max_epochs, out_dir, resume=False):
    os.makedirs(out_dir, exist_ok=True)
    last_path = os.path.join(out_dir, "last.pt")

    log_path = os.path.join(out_dir, "log.csv")
    if not resume and os.path.exists(log_path):
        os.remove(log_path)   

    patience = max(1, int(0.2 * max_epochs))
    best_score = -1.0
    epochs_without_improvement = 0
    start_epoch = 1

    if resume and os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        epochs_without_improvement = checkpoint["epochs_without_improvement"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = validate(model, val_loader, device)
        print(f"epoch {epoch}/{max_epochs}  train loss {train_loss:.4f}  "
              f"val score {val['score']:.4f} (loc {val['f1_loc']:.4f}, dmg {val['f1_dmg']:.4f})")

        append_log(log_path, {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_score": val["score"],
            "f1_loc": val["f1_loc"],
            "f1_dmg": val["f1_dmg"],
            "f1_no_damage": val["f1_per_class"][1],
            "f1_minor": val["f1_per_class"][2],
            "f1_major": val["f1_per_class"][3],
            "f1_destroyed": val["f1_per_class"][4],
        })

        if val["score"] > best_score:
            best_score = val["score"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            print("  -> new best model saved")
        else:
            epochs_without_improvement += 1

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
        }, last_path)
            
        if epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break

    return best_score