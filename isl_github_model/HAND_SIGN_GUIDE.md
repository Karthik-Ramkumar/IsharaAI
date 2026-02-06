# Indian Sign Language (ISL) Hand Sign Reference

## Tips for Better Accuracy

### 1. **Lighting**
- Use good, even lighting on your hand
- Avoid backlighting or shadows

### 2. **Hand Position**
- Keep your hand centered in the frame
- Maintain consistent distance from camera (about arm's length)
- Show your palm clearly to the camera
- Keep hand steady - the model needs 8 stable frames to confirm

### 3. **Background**
- Use a plain, contrasting background
- Avoid cluttered or busy backgrounds

### 4. **Sign Execution**
- Hold each sign clearly and steadily
- Wait for "STABLE" indicator (green text)
- Some signs look similar - check top 3 predictions

## Common Issues

### If accuracy is low (50-60%):
1. **Check Top 3 Predictions** - Your sign might be in the top 3
2. **Hold Steady** - Model needs 8 consecutive frames with same prediction
3. **Increase Confidence** - Edit `confidence_threshold` in code (currently 0.70 = 70%)
4. **Reduce Stability Frames** - Edit `stability_frames` in code (currently 8)

### Confusing Signs:
Some letters look similar in ISL:
- **M, N** - Finger positions are subtle
- **S, T** - Thumb position differs
- **U, V** - Finger separation
- **K, P** - Hand orientation matters

## Adjusting Settings

In `run_detector_stable.py`, you can adjust:

```python
stability_frames = 8      # Lower = faster but less stable (try 5-6)
confidence_threshold = 0.70  # Lower = more detections (try 0.60)
```

**Lower values = faster detection but more errors**
**Higher values = more accurate but slower**

## Testing Your Signs

1. Run the stable detector
2. Make a sign and hold it steady
3. Watch the "Top Predictions" box (right side)
4. If your intended sign appears in top 3, your hand position is close
5. Adjust your hand based on what the model is seeing
6. Wait for "STABLE" indicator before pressing SPACE

## Model Limitations

This model was trained on a specific ISL dataset. It recognizes:
- Numbers: 1-9 (no 0)
- Letters: A-Z

The model expects hand landmarks in a specific format. If signs don't match the training data exactly, accuracy will drop.
