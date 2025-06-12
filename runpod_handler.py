import os
import runpod
import tempfile
import requests
from dice_talk import DICE_Talk
import uuid

# Initialize the DICE_Talk pipe
# It's recommended to initialize models outside the handler function
# to reuse them across multiple requests if the worker stays warm.
pipe = DICE_Talk(device_id=0) # Assuming GPU device_id 0

def download_file(url, local_path):
    """Downloads a file from a URL to a local path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def handler(job):
    """
    Handler function for RunPod Serverless.
    Takes a job input, processes it with DICE_Talk, and returns the output.
    """
    job_input = job['input']

    image_url = job_input.get('image_url')
    audio_url = job_input.get('audio_url')
    emotion_type = job_input.get('emotion', 'neutral') # Default to neutral if not specified

    # Optional parameters from demo.py, with defaults
    ref_scale = float(job_input.get('ref_scale', 3.0))
    emo_scale = float(job_input.get('emo_scale', 6.0))
    crop = bool(job_input.get('crop', False)) # Input should be true or false
    seed = job_input.get('seed')
    if seed is not None:
        seed = int(seed)

    if not image_url or not audio_url:
        return {"error": "Missing image_url or audio_url in input"}

    # Create temporary files for downloaded inputs and output
    # Using a temporary directory that cleans up after itself
    with tempfile.TemporaryDirectory() as tmpdir:
        local_image_path = os.path.join(tmpdir, f"input_image_{uuid.uuid4().hex}.png") # Assume png, or try to infer
        local_audio_path = os.path.join(tmpdir, f"input_audio_{uuid.uuid4().hex}.wav") # Assume wav, or try to infer
        local_output_path = os.path.join(tmpdir, f"output_video_{uuid.uuid4().hex}.mp4")

        # Download image and audio
        print(f"Downloading image from: {image_url}")
        if not download_file(image_url, local_image_path):
            return {"error": f"Failed to download image from {image_url}"}

        print(f"Downloading audio from: {audio_url}")
        if not download_file(audio_url, local_audio_path):
            return {"error": f"Failed to download audio from {audio_url}"}

        # Determine emotion file path
        # Ensure this path matches the structure in your Docker image
        emotion_file_name = f"{emotion_type}.npy"
        emotion_path = os.path.join("examples", "emo", emotion_file_name)

        if not os.path.exists(emotion_path):
            # Fallback to neutral if specified emotion file doesn't exist
            print(f"Emotion file {emotion_path} not found. Falling back to neutral.")
            emotion_path = os.path.join("examples", "emo", "neutral.npy")
            if not os.path.exists(emotion_path):
                 return {"error": f"Neutral emotion file not found at {emotion_path}. Critical emotion files missing."}


        print(f"Processing with DICE-Talk: image='{local_image_path}', audio='{local_audio_path}', emotion='{emotion_path}'")

        try:
            # Preprocessing step (cropping if specified)
            face_info = pipe.preprocess(local_image_path, expand_ratio=0.5)
            print(f"Face info: {face_info}")

            processed_image_path = local_image_path
            if face_info['face_num'] > 0: # Check if face_num is greater than 0, not >=0
                if crop:
                    crop_image_filename = os.path.join(tmpdir, f"input_image_cropped_{uuid.uuid4().hex}.png")
                    pipe.crop_image(local_image_path, crop_image_filename, face_info['crop_bbox'])
                    processed_image_path = crop_image_filename
                    print(f"Cropped image saved to: {processed_image_path}")
            elif face_info['face_num'] == 0:
                print("No face detected in the image. Proceeding with the original image.")
            else: # face_num < 0, indicates an error during preprocessing
                return {"error": f"Error during face preprocessing: {face_info.get('error_message', 'Unknown error')}"}


            # Main processing
            pipe.process(
                image_path=processed_image_path,
                audio_path=local_audio_path,
                emotion_path=emotion_path,
                output_path=local_output_path,
                min_resolution=512, # Default from demo.py
                inference_steps=25, # Default from demo.py
                ref_scale=ref_scale,
                emo_scale=emo_scale,
                seed=seed
            )

            if os.path.exists(local_output_path):
                # Upload the video to a bucket or return as a presigned URL
                # For simplicity here, we'll return the video data as a base64 string
                # or a URL if using runpod.upload_file (which requires bucket setup)

                # Example: Upload to RunPod's temporary S3 bucket and get a presigned URL
                # This is generally preferred for larger files.
                # presigned_url = runpod.upload_file(f"output_{job['id']}.mp4", local_output_path)
                # return {"video_url": presigned_url}

                # For now, let's assume the video might be small enough to be returned directly
                # or this part needs to be configured by the user with their S3 bucket.
                # As a placeholder, we'll indicate success and path.
                # In a real scenario, you'd upload this file and return the URL.
                print(f"Output video successfully generated: {local_output_path}")
                # To return the actual file, you'd need to read and encode it, or use runpod.upload
                # For testing, we can just return a success message.
                # For actual use, the user needs to decide how to handle the output file (e.g., upload to S3)
                # For now, we'll just confirm it was created. RunPod itself might have a way to retrieve it from tmp.

                # Returning the file content directly (might hit size limits for large videos)
                with open(local_output_path, 'rb') as video_file:
                    video_data = video_file.read()

                # runpod.serverless.utils.rp_upload.upload_file_to_bucket() can be used if S3 integrated
                # For now, returning a success message, actual file handling might need more setup
                # by the user (e.g. setting up an S3 bucket for runpod.upload)

                # The simplest way for a user to test without S3 is to get it from worker logs or have a very small output.
                # A better approach is to upload it and return a URL.
                # For now, let's make it try to upload to job-specific output if available.
                # This is a RunPod feature that allows direct output of files.
                return {"output_video_path": local_output_path, "message": "Video generated. How to retrieve it depends on RunPod output configuration."}


            else:
                return {"error": "Output video not generated."}
        except Exception as e:
            print(f"Error during DICE-Talk processing: {e}")
            return {"error": f"Error during DICE-Talk processing: {str(e)}"}

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
