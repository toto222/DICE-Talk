import os
import runpod
import tempfile
import requests # Make sure requests is imported
from dice_talk import DICE_Talk
import uuid

# BunnyCDN Configuration (Consider moving AccessKey to environment variables for security in a real scenario)
BUNNYCDN_STORAGE_HOSTNAME = "storage.bunnycdn.com"
BUNNYCDN_STORAGE_ZONE_NAME = "zockto" # Your storage zone name
BUNNYCDN_VIDEO_PATH = "video" # The path within your storage zone
BUNNYCDN_ACCESS_KEY = "17e23633-2a7a-4d29-9450be4d6c8e-e01f-45f4" # Your AccessKey
BUNNYCDN_PUBLIC_HOSTNAME = "zockto.b-cdn.net" # Your public CDN hostname

pipe = DICE_Talk(device_id=0)

def download_file(url, local_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None

def upload_to_bunnycdn(local_file_path, unique_key):
    """Uploads a file to BunnyCDN storage and returns the public URL."""
    file_name = f"{unique_key}.mp4"
    upload_url = f"https://{BUNNYCDN_STORAGE_HOSTNAME}/{BUNNYCDN_STORAGE_ZONE_NAME}/{BUNNYCDN_VIDEO_PATH}/{file_name}"

    headers = {
        "AccessKey": BUNNYCDN_ACCESS_KEY,
        "Content-Type": "application/octet-stream" # Or video/mp4, but octet-stream is common for PUT
    }

    try:
        with open(local_file_path, 'rb') as f_data:
            response = requests.put(upload_url, data=f_data, headers=headers)

        response.raise_for_status() # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

        if response.status_code == 201: # Created
            public_url = f"https://{BUNNYCDN_PUBLIC_HOSTNAME}/{BUNNYCDN_VIDEO_PATH}/{file_name}"
            return True, public_url
        else:
            error_message = f"BunnyCDN upload failed with status {response.status_code}: {response.text}"
            print(error_message)
            return False, error_message

    except requests.exceptions.RequestException as e:
        error_message = f"Error uploading to BunnyCDN: {e}"
        print(error_message)
        return False, error_message
    except IOError as e:
        error_message = f"Error reading local file {local_file_path}: {e}"
        print(error_message)
        return False, error_message


def handler(job):
    job_input = job['input']

    image_url = job_input.get('image_url')
    audio_url = job_input.get('audio_url')
    emotion_type = job_input.get('emotion', 'neutral')

    ref_scale = float(job_input.get('ref_scale', 3.0))
    emo_scale = float(job_input.get('emo_scale', 6.0))
    crop = bool(job_input.get('crop', False))
    seed = job_input.get('seed')
    if seed is not None:
        seed = int(seed)

    if not image_url or not audio_url:
        return {"error": "Missing image_url or audio_url in input"}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate unique names for temporary local files
        temp_image_filename = f"input_image_{uuid.uuid4().hex}"
        temp_audio_filename = f"input_audio_{uuid.uuid4().hex}"
        # Try to get extension from URL or default
        image_ext = os.path.splitext(image_url)[1] or '.png'
        audio_ext = os.path.splitext(audio_url)[1] or '.wav'

        local_image_path = os.path.join(tmpdir, f"{temp_image_filename}{image_ext}")
        local_audio_path = os.path.join(tmpdir, f"{temp_audio_filename}{audio_ext}")
        # Output video will also be temporary before upload
        local_output_path = os.path.join(tmpdir, f"output_video_{uuid.uuid4().hex}.mp4")


        print(f"Downloading image from: {image_url} to {local_image_path}")
        if not download_file(image_url, local_image_path):
            return {"error": f"Failed to download image from {image_url}"}

        print(f"Downloading audio from: {audio_url} to {local_audio_path}")
        if not download_file(audio_url, local_audio_path):
            return {"error": f"Failed to download audio from {audio_url}"}

        emotion_file_name = f"{emotion_type}.npy"
        emotion_path = os.path.join("examples", "emo", emotion_file_name)

        if not os.path.exists(emotion_path):
            print(f"Emotion file {emotion_path} not found. Falling back to neutral.")
            emotion_path = os.path.join("examples", "emo", "neutral.npy")
            if not os.path.exists(emotion_path):
                 return {"error": f"Neutral emotion file not found at {emotion_path}. Critical emotion files missing."}

        print(f"Processing with DICE-Talk: image='{local_image_path}', audio='{local_audio_path}', emotion='{emotion_path}'")

        try:
            face_info = pipe.preprocess(local_image_path, expand_ratio=0.5)
            print(f"Face info: {face_info}")

            processed_image_path = local_image_path
            if face_info['face_num'] > 0:
                if crop:
                    crop_image_filename = os.path.join(tmpdir, f"input_image_cropped_{uuid.uuid4().hex}.png")
                    pipe.crop_image(local_image_path, crop_image_filename, face_info['crop_bbox'])
                    processed_image_path = crop_image_filename
                    print(f"Cropped image saved to: {processed_image_path}")
            elif face_info['face_num'] == 0:
                print("No face detected in the image. Proceeding with the original image.")
            else:
                return {"error": f"Error during face preprocessing: {face_info.get('error_message', 'Unknown error')}"}

            pipe.process(
                image_path=processed_image_path,
                audio_path=local_audio_path,
                emotion_path=emotion_path,
                output_path=local_output_path,
                min_resolution=512,
                inference_steps=25,
                ref_scale=ref_scale,
                emo_scale=emo_scale,
                seed=seed
            )

            if os.path.exists(local_output_path):
                print(f"Output video successfully generated: {local_output_path}")

                video_unique_key = str(uuid.uuid4())
                upload_success, result_or_error = upload_to_bunnycdn(local_output_path, video_unique_key)

                if upload_success:
                    public_video_url = result_or_error
                    print(f"Video uploaded to BunnyCDN: {public_video_url}")
                    return {"video_url": public_video_url}
                else:
                    error_message = result_or_error
                    print(f"Failed to upload video to BunnyCDN: {error_message}")
                    return {"error": f"Failed to upload video to BunnyCDN: {error_message}"}
            else:
                return {"error": "Output video not generated by DICE-Talk process."}
        except Exception as e:
            print(f"Error during DICE-Talk processing or upload: {e}")
            # It's good to log the stack trace for debugging
            import traceback
            traceback.print_exc()
            return {"error": f"Error during processing or upload: {str(e)}"}

runpod.serverless.start({"handler": handler})
