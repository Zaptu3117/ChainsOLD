import os
import subprocess

def display_mermaid(graph, name = "", open_file = False):
    try:
        # Assuming graph.get_graph().draw_mermaid_png() returns the image data
        image_data = graph.get_graph().draw_mermaid_png()
        
        # Save the image to a file
        with open(f'{name}_mermaid_diagram.png', 'wb') as f:
            f.write(image_data)
        
        # Open the image with the default viewer
        if open_file == True:  # For Windows
            os.startfile(f'{name}_mermaid_diagram.png')

    except Exception as e:
        print(f"Failed to display Mermaid diagram: {e}")