import csv
import pymap3d as pm
import numpy as np

def extract_enu_telemetry(csv_path, output_path):
    enu_coordinates = []
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        
        # Initialize the local origin using the very first coordinate
        lat0, lon0, alt0 = None, None, None
        
        for row in reader:
            # Adjust keys based on your specific drone's CSV/SRT output
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            alt = float(row['altitude'])
            timestamp = row['timestamp']
            
            # Set origin to the first frame's position
            if lat0 is None:
                lat0, lon0, alt0 = lat, lon, alt
            
            # Convert Geodetic (WGS84) to Local ENU (meters)
            e, n, u = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
            
            enu_coordinates.append({
                'timestamp': timestamp,
                'east': e,
                'north': n,
                'up': u
            })
            
    # Save the Cartesian coordinates for SfM Bundle Adjustment
    with open(output_path, mode='w', newline='') as outfile:
        fieldnames = ['timestamp', 'east', 'north', 'up']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enu_coordinates)
        
    print(f"Converted {len(enu_coordinates)} coordinates to local ENU.")
    print(f"Origin Anchor (Lat, Lon, Alt): {lat0}, {lon0}, {alt0}")

# Execution
# Assuming 'flight_log.csv' contains 'timestamp', 'latitude', 'longitude', 'altitude' columns
extract_enu_telemetry('flight_log.csv', 'enu_telemetry.csv')