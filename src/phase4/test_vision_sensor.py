"""
Quick test: Verify Camera sensor works and visualize vision features

Tests:
1. Camera sensor attachment and image capture
2. Vision processing (_process_vision method)
3. Visual output showing detected edges and zones
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import Camera

# Check for OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("❌ OpenCV not installed! Install with: pip install opencv-python")
    sys.exit(1)

def test_camera_vision():
    """Test camera sensor and vision processing"""
    
    print("=" * 60)
    print("PHASE 4C: Vision Sensor Test")
    print("=" * 60)
    
    # Setup logging
    set_up_simple_logging()
    
    # Connect to BeamNG (assume already running)
    print("\n1. Connecting to BeamNG...")
    bng_home = "S:/SteamLibrary/steamapps/common/BeamNG.drive"
    
    try:
        # Try connecting to existing instance
        bng = BeamNGpy('localhost', 64256)
        bng.open(launch=False)
        print("✓ Connected to existing BeamNG instance")
    except:
        print("No running instance found, launching BeamNG...")
        bng = BeamNGpy('localhost', 64256, home=bng_home)
        bng.open(launch=True)
        print("✓ BeamNG launched")
    
    # Setup scenario
    print("\n2. Setting up scenario...")
    scenario = Scenario('west_coast_usa', 'vision_test', description='Camera Vision Test')
    vehicle = Vehicle('test_car', model='etk800', license='VISION')
    
    # Attach camera (Camera uses old API style with name, bng, vehicle)
    camera = Camera('test_cam', bng, vehicle,
                   pos=(0, 1.5, 1.0), dir=(0, 1, 0), up=(0, 0, 1),
                   resolution=(480, 270), field_of_view_y=70,
                   is_render_colours=True, is_render_annotations=False,
                   is_render_instance=False, is_render_depth=False)
    print("✓ Camera created (480x270, 70° FOV)")
    
    # Spawn at highway position
    spawn_pos = (-717.121, 101, 118.675)
    spawn_rot = (0, 0, 0.3826834, 0.9238795)
    scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot)
    
    scenario.make(bng)
    bng.settings.set_deterministic(60)
    
    print("\n3. Loading scenario...")
    bng.scenario.load(scenario)
    bng.scenario.start()
    
    print("Waiting for physics to stabilize...")
    time.sleep(3)
    
    # Release parking brake
    vehicle.control(parkingbrake=0, throttle=0.5)
    time.sleep(1)
    
    # Test camera capture and vision processing
    print("\n4. Testing camera and vision processing...")
    
    for i in range(5):
        print(f"\n--- Frame {i+1}/5 ---")
        
        # Poll sensors
        vehicle.sensors.poll()
        
        # Get camera data
        camera_data = vehicle.sensors['camera']
        
        if camera_data is None or 'colour' not in camera_data:
            print("❌ No camera data received!")
            continue
        
        # Process image
        img = np.array(camera_data['colour'], dtype=np.uint8)
        print(f"✓ Camera image captured: {img.shape}")
        
        # Downsample
        img_small = cv2.resize(img, (240, 135))
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Zone analysis
        h, w = edges.shape
        zone_width = w // 3
        
        left_zone = edges[:, :zone_width]
        center_zone = edges[:, zone_width:2*zone_width]
        right_zone = edges[:, 2*zone_width:]
        
        left_edges = np.sum(left_zone > 0) / left_zone.size
        center_edges = np.sum(center_zone > 0) / center_zone.size
        right_edges = np.sum(right_zone > 0) / right_zone.size
        
        left_clear = 1.0 - np.clip(left_edges * 3, 0, 1)
        center_clear = 1.0 - np.clip(center_edges * 3, 0, 1)
        right_clear = 1.0 - np.clip(right_edges * 3, 0, 1)
        
        # Road edge detection
        hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
        road_mask = cv2.inRange(hsv[:, :, 2], 0, 100)
        
        bottom_half = road_mask[h//2:, :]
        road_left_edge = 0.5
        road_right_edge = 0.5
        
        for row in bottom_half:
            road_pixels = np.where(row > 0)[0]
            if len(road_pixels) > 10:
                leftmost = road_pixels[0]
                rightmost = road_pixels[-1]
                road_left_edge = leftmost / w
                road_right_edge = (w - rightmost) / w
                break
        
        print(f"Vision Features:")
        print(f"  Left Clear:   {left_clear:.2f} (1.0=clear, 0.0=obstacles)")
        print(f"  Center Clear: {center_clear:.2f}")
        print(f"  Right Clear:  {right_clear:.2f}")
        print(f"  Road Left:    {road_left_edge:.2f} (distance to left edge)")
        print(f"  Road Right:   {road_right_edge:.2f} (distance to right edge)")
        
        # Visualize zones on image
        vis_img = img_small.copy()
        
        # Draw zone dividers
        cv2.line(vis_img, (zone_width, 0), (zone_width, h), (0, 255, 0), 2)
        cv2.line(vis_img, (2*zone_width, 0), (2*zone_width, h), (0, 255, 0), 2)
        
        # Zone labels
        cv2.putText(vis_img, f"L:{left_clear:.2f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(vis_img, f"C:{center_clear:.2f}", (zone_width + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(vis_img, f"R:{right_clear:.2f}", (2*zone_width + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Save visualization
        output_path = f"vision_test_frame_{i+1}.png"
        cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved visualization: {output_path}")
        
        # Move car forward a bit
        vehicle.control(throttle=0.7, steering=0, brake=0)
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("✓ Vision sensor test complete!")
    print("Check vision_test_frame_*.png for visualizations")
    print("=" * 60)

if __name__ == '__main__':
    test_camera_vision()
