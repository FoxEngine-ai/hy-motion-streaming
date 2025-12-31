# 🎬 HY-Motion 1.0 Streaming Interface

This document describes the streaming Gradio interface that provides real-time frame-by-frame motion generation.

## Overview

The streaming interface allows you to watch motion generation in real-time, frame by frame, rather than waiting for the complete motion to be generated before seeing any results.

## Key Features

- **Real-time Generation**: See motion frames as they're generated
- **Progress Tracking**: Visual progress bar showing generation status
- **Interactive Control**: Stop generation at any time
- **Frame-by-Frame Updates**: Watch the motion evolve progressively
- **Same Quality**: Final result is identical to non-streaming version

## How It Works

### Technical Implementation

1. **Streaming ODE Solver**: Instead of computing the entire trajectory at once, the ODE solver yields intermediate results
2. **Frame Queue**: Generated frames are placed in a queue for smooth display
3. **Threaded Generation**: Generation happens in a separate thread to keep the UI responsive
4. **Progressive Decoding**: Motion data is decoded frame by frame
5. **Real-time HTML Updates**: The interface updates with each new frame

### Generation Process

```
Text Prompt → Streaming ODE Integration → Frame-by-Frame Decoding → Real-time Display
```

## Usage

### Starting the Streaming Interface

```bash
# Using Makefile (recommended)
make run-gradio-streaming

# Direct command
USE_HF_MODELS=0 python3 gradio_app_streaming.py

# With custom port
USE_HF_MODELS=0 python3 gradio_app_streaming.py --port 7862
```

### Interface Controls

- **📝 Input Text**: Enter your motion description
- **⏱️ Action Duration**: Set motion length (0.5-12 seconds)
- **🎲 Random Seeds**: Control randomness (comma-separated)
- **🎨 CFG Scale**: Control prompt adherence (1.0-10.0)
- **🚀 Generate Motion (Streaming)**: Start frame-by-frame generation
- **⏹️ Stop Generation**: Stop the streaming process

### What to Expect

1. **Initialization**: "Starting streaming generation..." message
2. **Frame Generation**: Progress bar shows advancement
3. **Real-time Updates**: Watch frames appear one by one
4. **Completion**: Full motion visualization appears when done

## Comparison: Streaming vs. Standard

| Feature | Streaming Interface | Standard Interface |
|---------|-------------------|-------------------|
| **Generation Time** | Same total time | Same total time |
| **User Experience** | See progress in real-time | Wait for completion |
| **Feedback** | Immediate visual feedback | Only final result |
| **Interactivity** | Can stop mid-generation | Must wait for completion |
| **Resource Usage** | Slightly higher (threading) | Standard |
| **Final Quality** | Identical | Identical |

## Technical Details

### Streaming Algorithm

```python
def generate_motion_streaming(self, text, seeds, duration, cfg_scale):
    # 1. Initialize ODE solver with parameters
    t = torch.linspace(0, 1, validation_steps + 1)
    
    # 2. Stream each time step
    for i, t_step in enumerate(t):
        if i > 0:  # Skip initial condition
            # Compute next frame
            current_x = integrate_ode_step(fn, current_x, t_step)
            
            # Decode intermediate motion
            intermediate_motion = decode_motion(current_x)
            
            # Yield frame data
            yield {
                "frame_index": i,
                "progress": i / total_steps,
                "motion_data": intermediate_motion
            }
            
            # Check for stop signal
            if stop_requested:
                break
    
    # 3. Final result with smoothing
    if not stopped:
        final_motion = decode_motion_with_smoothing(current_x)
        yield {"completed": True, "motion_data": final_motion}
```

### Performance Considerations

- **Memory**: Slightly higher memory usage due to intermediate frame storage
- **CPU**: Additional overhead from threading and queue management
- **GPU**: Same GPU utilization as standard generation
- **Network**: No additional network overhead

## Advanced Usage

### Custom Port

```bash
python3 gradio_app_streaming.py --port 8080
```

### Public Sharing

```bash
python3 gradio_app_streaming.py --share
```

### Development Mode

```bash
# Run both interfaces simultaneously
make run-gradio &  # Standard interface on port 7860
make run-gradio-streaming &  # Streaming interface on port 7861
```

## Troubleshooting

### Issue: Streaming is slow
- **Cause**: Real-time updates add overhead
- **Solution**: This is expected behavior for visual feedback

### Issue: Frames appear choppy
- **Cause**: Network latency or high system load
- **Solution**: Try reducing browser load or using a simpler prompt

### Issue: Generation stops unexpectedly
- **Cause**: Stop button clicked or error occurred
- **Solution**: Check status messages and try again

### Issue: Final result different from streaming preview
- **Cause**: Final result includes smoothing that's skipped during streaming
- **Solution**: This is normal - streaming shows raw frames, final shows smoothed result

## Limitations

1. **No FBX Streaming**: FBX files are only generated at the end
2. **Smoothing Delay**: Full smoothing is applied only to the final result
3. **Memory Usage**: Intermediate frames consume additional memory
4. **Complexity**: More complex than standard generation

## Future Enhancements

Potential improvements for future versions:
- **Adaptive Frame Rate**: Adjust streaming speed based on system performance
- **Frame Caching**: Allow pausing and resuming generation
- **Multi-User Streaming**: Support multiple simultaneous streaming sessions
- **WebSocket Support**: True real-time streaming without polling
- **Frame-by-Frame Export**: Save intermediate frames for analysis

## Integration with Makefile

The streaming interface is fully integrated with the project's Makefile:

```bash
# Build and run everything
make setup
make download-models
make run-gradio-streaming
```

## Support

For issues with the streaming interface:
1. Check the main README for general troubleshooting
2. Verify Python and dependency versions
3. Ensure GPU drivers are up to date
4. Try the standard interface first to isolate issues

The streaming interface provides an exciting way to experience motion generation in real-time while maintaining the same high-quality results as the standard interface!