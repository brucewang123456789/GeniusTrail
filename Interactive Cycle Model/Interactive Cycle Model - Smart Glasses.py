class DataTransmission:
    def __init__(self):
        print("Initializing Data Transmission module...")

    def transmit(self, text_data):
        try:
            print(f"Transmitting data: {text_data}")
            # Simulate data transmission
            transmitted_data = f"{text_data} transmitted"
            return transmitted_data
        except Exception as e:
            raise RuntimeError(f"Error in data transmission: {str(e)}")

class DataParsing:
    def __init__(self):
        print("Initializing Data Parsing module...")

    def parse(self, transmitted_data):
        try:
            print(f"Parsing transmitted data: {transmitted_data}")
            # Simulate parsing the transmitted data
            parsed_data = transmitted_data.replace(" transmitted", "")
            return parsed_data
        except Exception as e:
            raise RuntimeError(f"Error in data parsing: {str(e)}")

class TextFormatting:
    def __init__(self):
        print("Initializing Text Formatting module...")

    def format(self, parsed_data):
        try:
            print(f"Formatting parsed data: {parsed_data}")
            # Simulate text formatting (e.g., adjusting font size, alignment)
            formatted_text = f"[Formatted]: {parsed_data}"
            return formatted_text
        except Exception as e:
            raise RuntimeError(f"Error in text formatting: {str(e)}")

class TextRendering:
    def __init__(self):
        print("Initializing Text Rendering module...")

    def render(self, formatted_text):
        try:
            print(f"Rendering text: {formatted_text}")
            # Simulate text rendering for display
            rendered_text = f"[Rendered]: {formatted_text}"
            return rendered_text
        except Exception as e:
            raise RuntimeError(f"Error in text rendering: {str(e)}")

class Display:
    def __init__(self):
        print("Initializing Display module...")

    def show(self, rendered_text):
        try:
            print(f"Displaying on Smart Glasses: {rendered_text}")
            # Simulate displaying the rendered text
            display_output = f"[Displayed]: {rendered_text}"
            return display_output
        except Exception as e:
            raise RuntimeError(f"Error in display: {str(e)}")

class SmartGlassesPipeline:
    def __init__(self):
        self.data_transmission = DataTransmission()
        self.data_parsing = DataParsing()
        self.text_formatting = TextFormatting()
        self.text_rendering = TextRendering()
        self.display = Display()

    def run_pipeline(self, text_data):
        try:
            print("Running Smart Glasses pipeline...")
            transmitted_data = self.data_transmission.transmit(text_data)
            parsed_data = self.data_parsing.parse(transmitted_data)
            formatted_text = self.text_formatting.format(parsed_data)
            rendered_text = self.text_rendering.render(formatted_text)
            display_output = self.display.show(rendered_text)

            print("Smart Glasses pipeline completed.")
            return display_output
        except Exception as e:
            raise RuntimeError(f"Error in Smart Glasses pipeline: {str(e)}")

# Example Usage
if __name__ == "__main__":
    smart_glasses_pipeline = SmartGlassesPipeline()

    try:
        input_text = "Hello, this is a test message."  # Example input text
        result = smart_glasses_pipeline.run_pipeline(input_text)
        print(f"Final Smart Glasses Output: {result}")
    except Exception as e:
        print(f"Pipeline error: {e}")
