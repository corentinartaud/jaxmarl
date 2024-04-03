import json
import os


def save_json(dir, filename, obj):
  with open(os.path.join(dir, filename), "w") as f:
    json.dump(obj, f, sort_keys=True, indent=2)
    f.flush()


def log_metrics(dir, metrics_by_name, step):
  try:
    metrics_path = os.path.join(dir, "metrics.json")
    with open(metrics_path, "r") as f:
      saved_metrics = json.load(f)
  except IOError:
    # We haven't recorded anything yet. Start collecting.
    saved_metrics = {}
  
  for metric_name, metric_value in metrics_by_name.items():
    if metric_name not in saved_metrics:
      saved_metrics[metric_name] = []
    saved_metrics[metric_name] += [(step.item(), metric_value.item())]
  
  save_json(dir, "metrics.json", saved_metrics)
