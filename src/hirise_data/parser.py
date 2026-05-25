#!/usr/bin/env python3
"""
Filter rocks from satellite CSV based on size and position criteria.
"""
import csv
import argparse
import numpy as np

def filter_rocks(input_csv, output_csv, 
                 min_width=None, max_width=None,
                 min_length=None, max_length=None,
                 min_area=None, max_area=None,
                 x_min=None, x_max=None,
                 y_min=None, y_max=None,
                 center_x=None, center_y=None, radius=None):
    """
    Filter rocks from CSV based on criteria.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        min_width, max_width: Width range in meters
        min_length, max_length: Length range in meters
        min_area, max_area: Area range (width * length) in m²
        x_min, x_max: X position range in meters
        y_min, y_max: Y position range in meters
        center_x, center_y, radius: Circle selection (rocks within radius of center)
    """
    
    filtered_rocks = []
    total_count = 0
    
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_count += 1
            
            # Extract values
            x = float(row['Map_X'])
            y = float(row['Map_Y'])
            width = float(row['Width_m'])
            length = float(row['Length_m'])
            area = width * length
            print("testing")
            
            # Apply filters
            if min_width is not None and width < min_width:
                continue
            if max_width is not None and width > max_width:
                continue
                
            if min_length is not None and length < min_length:
                continue
            if max_length is not None and length > max_length:
                continue
                
            if min_area is not None and area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
                
            if x_min is not None and x < x_min:
                continue
            if x_max is not None and x > x_max:
                continue
                
            if y_min is not None and y < y_min:
                continue
            if y_max is not None and y > y_max:
                continue
                
            # Circle filter
            if center_x is not None and center_y is not None and radius is not None:
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist > radius:
                    continue
            
            # Passed all filters
            filtered_rocks.append(row)
    
    # Write output
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rocks)
    
    print(f"✅ Filtered {len(filtered_rocks)} / {total_count} rocks")
    print(f"   Saved to: {output_csv}")
    
    if len(filtered_rocks) > 0:
        # Print statistics
        widths = [float(r['Width_m']) for r in filtered_rocks]
        lengths = [float(r['Length_m']) for r in filtered_rocks]
        areas = [float(r['Width_m']) * float(r['Length_m']) for r in filtered_rocks]
        
        print(f"\n📊 Statistics:")
        print(f"   Width:  {min(widths):.2f} - {max(widths):.2f} m (avg: {np.mean(widths):.2f})")
        print(f"   Length: {min(lengths):.2f} - {max(lengths):.2f} m (avg: {np.mean(lengths):.2f})")
        print(f"   Area:   {min(areas):.3f} - {max(areas):.3f} m² (avg: {np.mean(areas):.3f})")


def main():
    parser = argparse.ArgumentParser(
        description='Filter rocks from satellite CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Rocks larger than 0.3m in any dimension
            python filter_rocks.py input.csv output.csv --min-width 0.3 --min-length 0.3
            
            # Rocks in a specific area
            python filter_rocks.py input.csv output.csv --x-min -15 --x-max -5 --y-min -25 --y-max -15
            
            # Rocks within 10m radius of origin
            python filter_rocks.py input.csv output.csv --center-x 0 --center-y 0 --radius 10
            
            # Small rocks only (area < 0.1 m²)
            python filter_rocks.py input.csv output.csv --max-area 0.1
                    """)
    
    parser.add_argument('input_csv', help='Input CSV file')
    parser.add_argument('output_csv', help='Output CSV file')
    
    # Size filters
    parser.add_argument('--min-width', type=float, help='Minimum width in meters')
    parser.add_argument('--max-width', type=float, help='Maximum width in meters')
    parser.add_argument('--min-length', type=float, help='Minimum length in meters')
    parser.add_argument('--max-length', type=float, help='Maximum length in meters')
    parser.add_argument('--min-area', type=float, help='Minimum area (width*length) in m²')
    parser.add_argument('--max-area', type=float, help='Maximum area (width*length) in m²')
    
    # Position filters (rectangle)
    parser.add_argument('--x-min', type=float, help='Minimum X position')
    parser.add_argument('--x-max', type=float, help='Maximum X position')
    parser.add_argument('--y-min', type=float, help='Minimum Y position')
    parser.add_argument('--y-max', type=float, help='Maximum Y position')
    
    # Position filters (circle)
    parser.add_argument('--center-x', type=float, help='Circle center X coordinate')
    parser.add_argument('--center-y', type=float, help='Circle center Y coordinate')
    parser.add_argument('--radius', type=float, help='Circle radius in meters')
    
    args = parser.parse_args()
    
    # Validate circle arguments
    circle_args = [args.center_x, args.center_y, args.radius]
    if any(x is not None for x in circle_args):
        if not all(x is not None for x in circle_args):
            parser.error("--center-x, --center-y, and --radius must all be specified together")
    
    filter_rocks(
        args.input_csv, args.output_csv,
        min_width=args.min_width, max_width=args.max_width,
        min_length=args.min_length, max_length=args.max_length,
        min_area=args.min_area, max_area=args.max_area,
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        center_x=args.center_x, center_y=args.center_y, radius=args.radius
    )


if __name__ == '__main__':
    main()