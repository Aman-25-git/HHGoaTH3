import os
import cv2
import json
import time
import hashlib
import requests
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

BASE_DIR = Path(__file__).resolve().parent
BLOCKCHAIN_FILE = BASE_DIR / "local_blockchain.json"

FACE_THRESHOLD = 0.363
MAX_RESULTS = 10

SERPAPI_IMAGE_URL = "https://serpapi.com/image"
SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Face Identification & Blockchain Verification",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>
    
    /* THIS CHANGES THE BACKGROUND COLOR */
    .stApp {
        background-color: #F0F8FF; 
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .success-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    .error-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e7f3ff;
        border: 1px solid #2196f3;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🔐 Face Identification & Blockchain Verification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">HH Goa 2026 — Town Hall 3</div>',
    unsafe_allow_html=True
)

st.info(
    "Pipeline: Face Scan → Face Detection → Face Encoding → "
    "Genuine Web Search → Matching Post → SHA-256 → Blockchain → Verification"
)

# ============================================================
# DOWNLOAD OPENCV AI MODELS
# ============================================================

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

YUNET_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"


YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def download_file(url, destination):

    if destination.exists():
        return True

    try:

        response = requests.get(
            url,
            timeout=60
        )

        response.raise_for_status()

        with open(destination, "wb") as f:
            f.write(response.content)

        return True

    except Exception as e:

        st.error(f"Model download failed: {e}")

        return False


# ============================================================
# LOAD FACE MODELS
# ============================================================

@st.cache_resource
def load_face_models():

    if not download_file(YUNET_URL, YUNET_MODEL):
        return None, None

    if not download_file(SFACE_URL, SFACE_MODEL):
        return None, None

    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL),
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(SFACE_MODEL),
        ""
    )

    return detector, recognizer


# ============================================================
# FACE DETECTION
# ============================================================

def detect_faces(image, detector):

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None:
        return []

    return faces


# ============================================================
# FACE EMBEDDING
# ============================================================

def get_face_embedding(image, face, recognizer):

    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    feature = recognizer.feature(
        aligned_face
    )

    return feature


# ============================================================
# FACE SIMILARITY
# ============================================================

def compare_faces(
    input_embedding,
    candidate_embedding,
    recognizer
):

    score = recognizer.match(
        input_embedding,
        candidate_embedding,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    return float(score)


# ============================================================
# LOCAL BLOCKCHAIN
# ============================================================

def load_blockchain():

    if not BLOCKCHAIN_FILE.exists():

        blockchain = {
            "chain": []
        }

        save_blockchain(blockchain)

    with open(
        BLOCKCHAIN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_blockchain(blockchain):

    with open(
        BLOCKCHAIN_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            blockchain,
            f,
            indent=4
        )


def calculate_sha256(data):

    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def calculate_block_hash(block):

    block_data = {
        "index": block["index"],
        "timestamp": block["timestamp"],
        "data_hash": block["data_hash"],
        "previous_hash": block["previous_hash"]
    }

    canonical = json.dumps(
        block_data,
        sort_keys=True,
        separators=(",", ":")
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def blockchain_add_record(data):

    blockchain = load_blockchain()

    data_hash = calculate_sha256(data)

    if len(blockchain["chain"]) == 0:

        previous_hash = "0" * 64
        index = 0

    else:

        previous_hash = blockchain["chain"][-1]["block_hash"]
        index = len(blockchain["chain"])

    block = {

        "index": index,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "data_hash": data_hash,

        "previous_hash": previous_hash
    }

    block["block_hash"] = calculate_block_hash(block)

    blockchain["chain"].append(block)

    save_blockchain(blockchain)

    return block


def blockchain_get_block(index):

    blockchain = load_blockchain()

    for block in blockchain["chain"]:

        if block["index"] == index:

            return block

    return None


def verify_blockchain_record(data, index):

    block = blockchain_get_block(index)

    if block is None:

        return False, None, None

    recalculated_hash = calculate_sha256(data)

    stored_hash = block["data_hash"]

    verified = (
        recalculated_hash == stored_hash
    )

    return (
        verified,
        stored_hash,
        recalculated_hash
    )


# ============================================================
# SERPAPI IMAGE UPLOAD
# ============================================================

def upload_image_to_serpapi(image_bytes):

    if not SERPAPI_KEY:

        raise ValueError(
            "SERPAPI_KEY is missing in .env"
        )

    files = {

        "image": (
            "face.jpg",
            image_bytes,
            "image/jpeg"
        )
    }

    data = {

        "api_key": SERPAPI_KEY
    }

    response = requests.post(
        SERPAPI_IMAGE_URL,
        files=files,
        data=data,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    if "error" in result:

        raise RuntimeError(
            result["error"]
        )

    if "image_id" not in result:

        raise RuntimeError(
            "SerpApi did not return image_id"
        )

    return result["image_id"]


# ============================================================
# GOOGLE LENS SEARCH
# ============================================================

def google_lens_search(image_id):

    params = {

        "engine": "google_lens",

        "image_id": image_id,

        "type": "exact_matches",

        "hl": "en",

        "country": "in",

        "safe": "active",

        "api_key": SERPAPI_KEY
    }

    response = requests.get(
        SERPAPI_SEARCH_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    if "error" in result:

        raise RuntimeError(
            result["error"]
        )

    return result


# ============================================================
# PARSE GOOGLE LENS RESULTS
# ============================================================

def extract_results(lens_result):

    candidates = []

    # Exact matches

    exact_matches = lens_result.get(
        "exact_matches",
        []
    )

    for item in exact_matches:

        candidates.append({

            "title": item.get(
                "title",
                "Untitled"
            ),

            "url": item.get(
                "link",
                ""
            ),

            "source": item.get(
                "source",
                ""
            ),

            "thumbnail": item.get(
                "thumbnail",
                ""
            ),

            "type": "exact_match"
        })


    # Visual matches

    visual_matches = lens_result.get(
        "visual_matches",
        []
    )

    for item in visual_matches:

        candidates.append({

            "title": item.get(
                "title",
                "Untitled"
            ),

            "url": item.get(
                "link",
                ""
            ),

            "source": item.get(
                "source",
                ""
            ),

            "thumbnail": item.get(
                "thumbnail",
                ""
            ),

            "type": "visual_match"
        })


    # Organic results

    organic_results = lens_result.get(
        "organic_results",
        []
    )

    for item in organic_results:

        candidates.append({

            "title": item.get(
                "title",
                "Untitled"
            ),

            "url": item.get(
                "link",
                ""
            ),

            "source": item.get(
                "source",
                ""
            ),

            "thumbnail": item.get(
                "thumbnail",
                ""
            ),

            "type": "organic"
        })


    # Remove duplicate URLs

    unique = {}

    for item in candidates:

        url = item["url"]

        if url and url not in unique:

            unique[url] = item


    return list(unique.values())[:MAX_RESULTS]


# ============================================================
# DOWNLOAD CANDIDATE IMAGE
# ============================================================

def download_candidate_image(url):

    try:

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent":
                "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if not content_type.startswith("image"):

            return None

        data = np.frombuffer(
            response.content,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception:

        return None


# ============================================================
# SEARCH RESULT FACE VERIFICATION
# ============================================================

def verify_candidate(
    candidate,
    input_embedding,
    detector,
    recognizer
):

    image_url = candidate.get(
        "thumbnail"
    )

    if not image_url:

        return None

    image = download_candidate_image(
        image_url
    )

    if image is None:

        return None

    faces = detect_faces(
        image,
        detector
    )

    if len(faces) == 0:

        return None

    best_score = 0.0

    for face in faces:

        try:

            embedding = get_face_embedding(
                image,
                face,
                recognizer
            )

            score = compare_faces(
                input_embedding,
                embedding,
                recognizer
            )

            best_score = max(
                best_score,
                score
            )

        except Exception:

            continue

    candidate["similarity"] = best_score

    return candidate


# ============================================================
# SEARCH PIPELINE
# ============================================================

def perform_web_search(
    image_bytes,
    input_embedding,
    detector,
    recognizer
):

    status = {}

    # Step 1

    status["upload"] = "Uploading image to reverse-image search..."

    image_id = upload_image_to_serpapi(
        image_bytes
    )

    # Step 2

    status["search"] = "Searching Google Lens..."

    lens_result = google_lens_search(
        image_id
    )

    # Step 3

    candidates = extract_results(
        lens_result
    )

    verified_candidates = []

    for candidate in candidates:

        result = verify_candidate(
            candidate,
            input_embedding,
            detector,
            recognizer
        )

        if result is not None:

            verified_candidates.append(
                result
            )

    verified_candidates.sort(
        key=lambda x: x.get(
            "similarity",
            0
        ),
        reverse=True
    )

    return (
        lens_result,
        verified_candidates
    )


# ============================================================
# STREAMLIT APP
# ============================================================

uploaded_file = st.file_uploader(
    "📸 Upload a face scan",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file:

    image_bytes = uploaded_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:

        st.error(
            "Invalid image."
        )

        st.stop()


    # ========================================================
    # FACE DETECTION
    # ========================================================

    st.header(
        "1️⃣ Face Identification"
    )

    with st.spinner(
        "Loading face recognition models..."
    ):

        detector, recognizer = load_face_models()


    if detector is None or recognizer is None:

        st.error(
            "Unable to load face models."
        )

        st.stop()


    faces = detect_faces(
        image,
        detector
    )


    col1, col2 = st.columns(2)


    with col1:

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Input Face Scan",
            use_container_width=True
        )


    with col2:

        st.metric(
            "Faces Detected",
            len(faces)
        )


    if len(faces) == 0:

        st.error(
            "❌ No face detected."
        )

        st.stop()


    if len(faces) > 1:

        st.warning(
            "Multiple faces detected. "
            "Using the first detected face."
        )


    input_face = faces[0]


    input_embedding = get_face_embedding(
        image,
        input_face,
        recognizer
    )


    st.success(
        "✅ Face detected and encoded successfully."
    )


    st.write(
        f"Embedding shape: {input_embedding.shape}"
    )


    # ========================================================
    # WEB SEARCH
    # ========================================================

    st.header(
        "2️⃣ Genuine Web / Social Media Search"
    )

    st.info(
        "This performs a runtime reverse-image search "
        "using Google Lens through SerpApi. "
        "Results are not hardcoded."
    )


    if not SERPAPI_KEY:

        st.error(
            "SERPAPI_KEY is not configured."
        )

        st.code(
            "SERPAPI_KEY=YOUR_KEY"
        )

        st.stop()


    if st.button(
        "🔎 Search Web for Matching Content",
        type="primary"
    ):

        try:

            with st.spinner(
                "Performing genuine reverse-image search..."
            ):

                (
                    lens_result,
                    verified_candidates
                ) = perform_web_search(
                    image_bytes,
                    input_embedding,
                    detector,
                    recognizer
                )


            st.session_state[
                "lens_result"
            ] = lens_result

            st.session_state[
                "candidates"
            ] = verified_candidates


            st.success(
                "✅ Genuine search completed."
            )


        except Exception as e:

            st.error(
                f"Search failed: {e}"
            )

            st.stop()


    candidates = st.session_state.get(
        "candidates",
        []
    )


    # ========================================================
    # DISPLAY SEARCH RESULTS
    # ========================================================

    if candidates:

        st.subheader(
            "🔍 Candidate Matches"
        )

        for index, candidate in enumerate(
            candidates
        ):

            similarity = candidate.get(
                "similarity",
                0
            )

            with st.expander(
                f"Candidate {index + 1}: "
                f"{candidate['title']}"
            ):

                c1, c2 = st.columns(
                    [1, 2]
                )

                with c1:

                    if candidate.get(
                        "thumbnail"
                    ):

                        st.image(
                            candidate["thumbnail"],
                            use_container_width=True
                        )

                with c2:

                    st.write(
                        "**Source:**",
                        candidate["source"]
                    )

                    st.write(
                        "**Type:**",
                        candidate["type"]
                    )

                    st.write(
                        "**Similarity:**",
                        f"{similarity:.4f}"
                    )

                    st.write(
                        "**URL:**",
                        candidate["url"]
                    )


        # Best candidate

        best_candidate = candidates[0]

        best_score = best_candidate.get(
            "similarity",
            0
        )


        st.header(
            "🎯 Best Matching Result"
        )


        if best_score >= FACE_THRESHOLD:

            st.markdown(
                '<div class="success-box">'
                '✅ MATCH FOUND'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"Similarity Score: "
                f"**{best_score:.4f}**"
            )

            st.write(
                "Title:",
                best_candidate["title"]
            )

            st.write(
                "Source:",
                best_candidate["source"]
            )

            st.write(
                "URL:",
                best_candidate["url"]
            )

            st.session_state[
                "best_candidate"
            ] = best_candidate


        else:

            st.markdown(
                '<div class="error-box">'
                '❌ NO VERIFIED FACE MATCH'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"Best similarity: "
                f"{best_score:.4f}"
            )

            st.write(
                f"Required threshold: "
                f"{FACE_THRESHOLD}"
            )

            st.stop()


    else:

        if "lens_result" in st.session_state:

            st.warning(
                "Google Lens returned results, "
                "but no candidate image could be "
                "face-verified."
            )


    # ========================================================
    # BLOCKCHAIN
    # ========================================================

    if "best_candidate" in st.session_state:

        st.header(
            "3️⃣ Blockchain Registration"
        )


        candidate = st.session_state[
            "best_candidate"
        ]


        blockchain_data = {

            "post_url": candidate["url"],

            "post_title": candidate["title"],

            "source": candidate["source"],

            "search_type": candidate["type"],

            "face_similarity": round(
                candidate["similarity"],
                6
            ),

            "registered_at": datetime.now(
                timezone.utc
            ).isoformat()
        }


        data_hash = calculate_sha256(
            blockchain_data
        )


        st.write(
            "**SHA-256 Fingerprint:**"
        )

        st.code(
            data_hash
        )


        if st.button(
            "⛓️ Register on Blockchain"
        ):

            try:

                block = blockchain_add_record(
                    blockchain_data
                )


                st.session_state[
                    "block"
                ] = block

                st.session_state[
                    "blockchain_data"
                ] = blockchain_data


                st.success(
                    "✅ Data registered on local blockchain."
                )


                st.write(
                    "**Block Number:**",
                    block["index"]
                )

                st.write(
                    "**Transaction / Block Hash:**",
                    block["block_hash"]
                )

                st.write(
                    "**Previous Hash:**",
                    block["previous_hash"]
                )

                st.write(
                    "**Timestamp:**",
                    block["timestamp"]
                )


            except Exception as e:

                st.error(
                    f"Blockchain registration failed: {e}"
                )


    # ========================================================
    # VERIFICATION
    # ========================================================

    if "block" in st.session_state:

        st.header(
            "4️⃣ Blockchain Verification"
        )


        if st.button(
            "🔐 Verify Blockchain Record",
            type="primary"
        ):

            block = st.session_state[
                "block"
            ]

            blockchain_data = st.session_state[
                "blockchain_data"
            ]


            verified, stored_hash, recalculated_hash = (
                verify_blockchain_record(
                    blockchain_data,
                    block["index"]
                )
            )


            st.write(
                "**Blockchain Hash:**"
            )

            st.code(
                stored_hash
            )


            st.write(
                "**Recalculated Hash:**"
            )

            st.code(
                recalculated_hash
            )


            if verified:

                st.markdown(
                    '<div class="success-box">'
                    '🔐 VERIFIED — DATA INTEGRITY CONFIRMED'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.success(
                    "The locally recalculated SHA-256 "
                    "matches the blockchain record."
                )

            else:

                st.markdown(
                    '<div class="error-box">'
                    '⚠️ TAMPERED — HASH MISMATCH'
                    '</div>',
                    unsafe_allow_html=True
                )


    # ========================================================
    # TAMPERING DEMO
    # ========================================================

    if "block" in st.session_state:

        st.header(
            "5️⃣ Tamper Detection Demonstration"
        )

        st.caption(
            "This changes one local field to demonstrate "
            "that the blockchain fingerprint detects modification."
        )


        if st.button(
            "🧪 Simulate Data Tampering"
        ):

            original_data = st.session_state[
                "blockchain_data"
            ].copy()


            tampered_data = original_data.copy()


            tampered_data[
                "post_title"
            ] = (
                original_data["post_title"]
                + " [MODIFIED]"
            )


            verified, stored_hash, recalculated_hash = (
                verify_blockchain_record(
                    tampered_data,
                    st.session_state[
                        "block"
                    ]["index"]
                )
            )


            st.write(
                "Original blockchain hash:"
            )

            st.code(
                stored_hash
            )


            st.write(
                "Hash after modification:"
            )

            st.code(
                recalculated_hash
            )


            if not verified:

                st.markdown(
                    '<div class="error-box">'
                    '⚠️ TAMPERING DETECTED'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.success(
                    "Blockchain verification correctly "
                    "detected that the data was modified."
                )

            else:

                st.error(
                    "Unexpected verification result."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HH Goa 2026 — Task 3 | "
    "Face Identification & Blockchain Verification"
)