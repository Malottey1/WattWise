#!/usr/bin/env python3
"""
Updated script to populate energy_readings table with corrected columns.
"""

import os
import sys
import csv
from datetime import datetime

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# Set environment variable for Flask
os.environ['FLASK_APP'] = 'app'
os.environ['FLASK_ENV'] = 'development'

try:
    from app import create_app, db
    from app.models import EnergyReading
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed")
    sys.exit(1)

def safe_float(value, default=None):
    """Safely convert to float, return default if invalid."""
    if not value or value.strip() in ['', '?', 'NA']:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def updated_populate():
    """Updated data population matching the new table structure."""
    
    print("🚀 Starting updated data population...")
    
    app = create_app()
    
    with app.app_context():
        # Check existing data
        existing_count = EnergyReading.query.count()
        if existing_count > 1000:
            print(f"⚠️ Database already has {existing_count} readings. Please clear first.")
            return
        
        # File path
        file_path = "app/household_power_consumption.txt"
        
        if not os.path.exists(file_path):
            print(f"❌ Data file not found at {file_path}")
            return
        
        print("📖 Reading data file with updated columns...")
        
        rows_added = 0
        max_rows = 200000
        batch_size = 1000
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                
                # Skip header
                header = next(reader)
                print(f"📋 File header: {header}")
                print(f"🔢 Expected columns: {len(header)}")
                
                batch_readings = []
                
                for row_num, row in enumerate(reader):
                    if rows_added >= max_rows:
                        break
                    
                    # Skip rows that don't have enough columns
                    if len(row) < 9:
                        if row_num < 10:  # Only show first few warnings
                            print(f"⚠️ Row {row_num}: insufficient columns ({len(row)}), skipping")
                        continue
                    
                    try:
                        # Parse date and time
                        date_str, time_str = row[0].strip(), row[1].strip()
                        timestamp = datetime.strptime(f"{date_str} {time_str}", '%d/%m/%Y %H:%M:%S')
                        
                        # Parse ALL values safely - matching the text file structure
                        global_active_power = safe_float(row[2])
                        global_reactive_power = safe_float(row[3])
                        voltage = safe_float(row[4])
                        global_intensity = safe_float(row[5])
                        sub_metering_1 = safe_float(row[6])
                        sub_metering_2 = safe_float(row[7])
                        sub_metering_3 = safe_float(row[8])
                        
                        # Skip if essential data is missing
                        if global_active_power is None or global_active_power <= 0:
                            continue
                        
                        # Create energy reading with UPDATED columns matching new table structure
                        reading = EnergyReading(
                            timestamp=timestamp,
                            global_active_power=global_active_power,
                            global_reactive_power=global_reactive_power,
                            voltage=voltage,
                            global_intensity=global_intensity,
                            sub_metering_1=sub_metering_1,
                            sub_metering_2=sub_metering_2,
                            sub_metering_3=sub_metering_3
                        )
                        
                        batch_readings.append(reading)
                        rows_added += 1
                        
                        # Commit in batches
                        if len(batch_readings) >= batch_size:
                            db.session.bulk_save_objects(batch_readings)
                            db.session.commit()
                            batch_readings = []
                            print(f"✅ Added {rows_added} rows...")
                    
                    except (ValueError, IndexError) as e:
                        if row_num < 10:  # Only show first few errors
                            print(f"❌ Row {row_num} error: {e}")
                        continue
                
                # Commit any remaining readings
                if batch_readings:
                    db.session.bulk_save_objects(batch_readings)
                    db.session.commit()
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"🎉 Data population complete! Added {rows_added} energy readings.")
        
        # Verify the data
        final_count = EnergyReading.query.count()
        print(f"📊 Total readings in database: {final_count}")
        
        # Show a sample of the data to verify all columns are populated
        if final_count > 0:
            sample = EnergyReading.query.first()
            print(f"📋 Sample reading verification:")
            print(f"   - Timestamp: {sample.timestamp}")
            print(f"   - Global Active Power: {sample.global_active_power}")
            print(f"   - Global Reactive Power: {sample.global_reactive_power}")
            print(f"   - Voltage: {sample.voltage}")
            print(f"   - Global Intensity: {sample.global_intensity}")
            print(f"   - Sub1: {sample.sub_metering_1}")
            print(f"   - Sub2: {sample.sub_metering_2}")
            print(f"   - Sub3: {sample.sub_metering_3}")

if __name__ == "__main__":
    updated_populate()