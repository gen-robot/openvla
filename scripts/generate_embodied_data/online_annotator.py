import os
import json
import numpy as np
from PIL import Image
import time
import random
try:
    from .utils import NumpyFloatValuesEncoder
    # Import visualization functions
    from .visualize_vlm_output import load_image, visualize_vlm_output
except:
    from utils import NumpyFloatValuesEncoder
    from visualize_vlm_output import load_image, visualize_vlm_output
from google import genai

class OnlineAnnotator:
    def __init__(self, model_name="gemini-2.5-pro-exp-0325", temperature=0.5, max_retries=3):
        """
        Initialize the online annotator with a specific model.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation (lower for more deterministic outputs)
            max_retries: Maximum number of retries for API calls
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.prompt_template = self._load_prompt_template()
        
        # Initialize token counters and query tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.query_count = 0
    
    def _load_prompt_template(self):
        """Load the VLM prompt template from file."""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "vlm_prompt1.txt")
        with open(prompt_path, "r") as f:
            return f.read()
    
    def _extract_json_from_response(self, response_text):
        """Extract JSON data from model response."""
        # Return None if response_text is None
        if response_text is None:
            print("Warning: Response text is None, cannot extract JSON")
            return None
            
        try:
            # Find the JSON content between ```json and ``` markers
            json_start = response_text.find('```json')
            if json_start == -1:
                # Try to find just the opening brace
                json_start = response_text.find('{')
            else:
                # Skip the ```json marker
                json_start = response_text.find('{', json_start)
            
            # If no opening brace found, return None
            if json_start == -1:
                print("Warning: No JSON content found in the response")
                return None
                
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Error extracting JSON from response: {e}")
            print(f"Response: {response_text}")
            return None
    
    def _rescale_coordinates(self, data, image_width, image_height):
        """Rescale coordinates from [0-1000] range to actual image dimensions."""
        # Clone the data to avoid modifying the original
        scaled_data = json.loads(json.dumps(data))
        
        # Rescale visible_objects boxes
        if "visible_objects" in scaled_data:
            for obj in scaled_data["visible_objects"]:
                if "box" in obj:
                    # Format is [tly, tlx, bry, brx]
                    tly, tlx, bry, brx = obj["box"]
                    obj["box"] = [
                        int(tly * image_height / 1000),
                        int(tlx * image_width / 1000),
                        int(bry * image_height / 1000),
                        int(brx * image_width / 1000)
                    ]
        
        # Rescale gripper position
        if "gripper_position" in scaled_data:
            gp = scaled_data["gripper_position"]
            
            # Current position
            if "current" in gp and gp["current"] != [-1, -1]:  # Only rescale if gripper is visible
                y, x = gp["current"]
                gp["current"] = [
                    int(y * image_height / 1000),
                    int(x * image_width / 1000)
                ]
            
            # Trajectory
            if "trajectory" in gp:
                for i, (y, x) in enumerate(gp["trajectory"]):
                    gp["trajectory"][i] = [
                        int(y * image_height / 1000),
                        int(x * image_width / 1000)
                    ]
        
        # Rescale key subgoals
        if "key_subgoals" in scaled_data:
            for subgoal in scaled_data["key_subgoals"]:
                if "location" in subgoal:
                    y, x = subgoal["location"]
                    subgoal["location"] = [
                        int(y * image_height / 1000),
                        int(x * image_width / 1000)
                    ]
        
        return scaled_data
    
    def _format_data_to_string(self, data):
        """
        Format the annotation data into a structured string format.
        
        Args:
            data: Processed JSON annotation data
            
        Returns:
            Formatted string representation of the annotation data
        """
        output_lines = []
        
        # Task
        if "task" in data:
            output_lines.append(f"TASK: {data['task']}")
        
        # Plan
        if "plan" in data:
            output_lines.append(f"PLAN: {data['plan']}")
        
        # Visible objects
        if "visible_objects" in data:
            objects_str = ", ".join([f"{obj['name']} {obj['box']}" for obj in data['visible_objects']])
            output_lines.append(f"VISIBLE OBJECTS: {objects_str}")
        
        # Subtask reasoning
        if "subtask_reasoning" in data:
            output_lines.append(f"SUBTASK REASONING: {data['subtask_reasoning']}")
        
        # Current subtask
        if "current_subtask" in data:
            output_lines.append(f"SUBTASK: {data['current_subtask']}")
        
        # Relevant objects
        if "relevant_objects" in data:
            rel_objs = [obj for obj in data["relevant_objects"]]
            output_lines.append(f"RELEVANT OBJECTS: {rel_objs}")
        
        # Move reasoning
        if "move_reasoning" in data:
            output_lines.append(f"MOVE REASONING: {data['move_reasoning']}")
        
        # Move command
        if "move" in data:
            output_lines.append(f"MOVE: {data['move']}")
        
        # Gripper position
        if "gripper_position" in data and "trajectory" in data["gripper_position"]:
            # Flatten the trajectory into a single list
            trajectory = data["gripper_position"]["trajectory"]
            flat_trajectory = [coord for point in trajectory for coord in point]
            output_lines.append(f"GRIPPER POSITION: {flat_trajectory}")
        
        return " ".join(output_lines)
    
    def generate_with_image(self, prompt, image):
        """Generate text response based on prompt and image input with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Convert numpy array to PIL Image if needed
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                # Use the direct client.models.generate_content pattern
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        image,
                        prompt
                    ],
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature
                    )
                )
                
                # Increment query count on successful call
                self.query_count += 1
                
                # Count tokens if available in the response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    input_tokens = 0
                    output_tokens = 0
                    
                    print("-" * 50)
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        input_tokens = response.usage_metadata.prompt_token_count
                        self.total_input_tokens += input_tokens
                        print(f"Input tokens: {input_tokens} (Total: {self.total_input_tokens}, Avg: {self.total_input_tokens/self.query_count:.1f} over {self.query_count} queries)")
                    
                    if hasattr(response.usage_metadata, 'candidates_token_count'):
                        output_tokens = response.usage_metadata.candidates_token_count
                        self.total_output_tokens += output_tokens
                        print(f"Output tokens: {output_tokens} (Total: {self.total_output_tokens}, Avg: {self.total_output_tokens/self.query_count:.1f} over {self.query_count} queries)")
                    
                    print(f"Total queries: {self.query_count}")
                
                if hasattr(response, 'text'):
                    return response.text
                else:
                    print("Warning: Response has no text attribute")
                    # Try to extract text from candidates if available
                    if hasattr(response, 'candidates') and response.candidates:
                        return response.candidates[0].content.parts[0].text
                    return None
            except Exception as e:
                print(f"Error generating with image (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    # Add a random jitter to the backoff time to prevent synchronized retries
                    backoff_time = 2 ** attempt + random.uniform(0, 1)
                    print(f"Retrying in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
                else:
                    print("Max retries reached, returning None")
                    return None
        return None
    
    def annotate(self, language_instruction, image, image_width=None, image_height=None, 
                visualize=False, output_path=None, show_visualization=False, 
                return_formatted_string=False):
        """
        Generate VLM annotations for the given image and language instruction.
        
        Args:
            language_instruction: The natural language instruction for the task
            image: PIL Image, numpy array, or path to image file
            image_width: Width of the image (optional if image is PIL or numpy)
            image_height: Height of the image (optional if image is PIL or numpy)
            visualize: Whether to visualize the annotations (default: False)
            output_path: Path to save visualization if visualize=True (optional)
            show_visualization: Whether to display the visualization (default: False)
            return_formatted_string: Whether to return a formatted string representation (default: False)
            
        Returns:
            Annotation data in JSON format with rescaled coordinates or formatted string if requested
        """
        # Prepare the prompt by replacing the placeholder
        prompt = self.prompt_template.replace("LANGUAGE_INSTRUCTION", language_instruction)
        
        # Get image dimensions if not provided
        if image_width is None or image_height is None:
            if isinstance(image, np.ndarray):
                image_height, image_width = image.shape[:2]
            elif isinstance(image, str):
                with Image.open(image) as img:
                    image_width, image_height = img.size
            elif isinstance(image, Image.Image):
                image_width, image_height = image.size
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Store original image path if it's a string
        image_path = image if isinstance(image, str) else None
        
        # Load image from path if a string was provided
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                print(f"Error loading image from path: {e}")
                return None
        
        # Query the model with image and prompt
        response_text = self.generate_with_image(prompt, image)
        
        # Extract and parse the JSON data
        json_data = self._extract_json_from_response(response_text)
        
        # If no valid data was extracted, return None
        if not json_data:
            print("Warning: Failed to extract valid annotation data")
            return None
            
        # Rescale coordinates to match actual image dimensions
        try:
            json_data = self._rescale_coordinates(json_data, image_width, image_height)
            
            # Visualize the annotations if requested
            if visualize:
                # If image_path is None, we need to temporarily save the image
                temp_image_path = None
                if image_path is None:
                    temp_image_path = "temp_image.png"
                    image.save(temp_image_path)
                    image_path = temp_image_path
                
                try:
                    # Load the image for visualization
                    img, img_width, img_height = load_image(image_path)
                    
                    # Set default output path if not provided
                    if output_path is None and not show_visualization:
                        output_path = f"vlm_visualization_{self.query_count}.png"
                    
                    # Visualize the annotations - specify normalized=False because coordinates are already rescaled
                    visualize_vlm_output(img, img_width, img_height, json_data, 
                                       output_path, show_visualization, normalized=False)
                except Exception as e:
                    print(f"Error during visualization: {e}")
                finally:
                    # Clean up temporary image if created
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
        except Exception as e:
            print(f"Error processing annotation data: {e}")
            return json_data  # Return the unprocessed data if rescaling fails
        
        # Return formatted string if requested
        if return_formatted_string and json_data:
            try:
                return self._format_data_to_string(json_data)
            except Exception as e:
                print(f"Error formatting data to string: {e}")
                return json_data
        
        return json_data
    
    def close(self):
        """Close the client to prevent resource warnings during garbage collection."""
        try:
            # Access to the internal _client to close it properly
            if hasattr(self.client, '_client') and self.client._client is not None:
                self.client._client.close()
            # For newer versions that might have different structure
            elif hasattr(self.client, 'close'):
                self.client.close()
        except Exception as e:
            print(f"Warning: Could not close client gracefully: {e}")
            
    def __del__(self):
        """Ensure client is closed when the annotator is garbage collected."""
        self.close()


if __name__ == "__main__":
    # Example usage with visualization
    annotator = OnlineAnnotator(model_name="gemini-2.5-pro-exp-03-25")

    # Use the annotator
    result = annotator.annotate(
        language_instruction="push the plate to the front of the stove",
        image="examples/example1.jpg",
        visualize=True,
        output_path="examples/push_the_plate_to_the_front_of_the_stove_ep229.png",
        return_formatted_string=False
    )
    
    # Save the result
    with open("push_the_plate_to_the_front_of_the_stove_ep229.json", 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyFloatValuesEncoder)
            
    # print(result)