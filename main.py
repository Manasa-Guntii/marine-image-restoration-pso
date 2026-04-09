import streamlit as st
import numpy as np
import cv2 as cv
import tempfile
from nuce_utils import *
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="NUCE Enhancement", layout="wide")

st.title("🌊 NUCE Underwater Image & Video Enhancement")

alpha = st.sidebar.slider("Sharpen Strength",0.0,0.5,0.2)

mode = st.radio(
"Select Input Type",
["Image Upload","Video Upload","Webcam Enhancement"]
)

# ---------------- IMAGE ----------------

if mode=="Image Upload":

    uploaded = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

    if uploaded:

        file_bytes = np.asarray(bytearray(uploaded.read()),dtype=np.uint8)
        img = cv.imdecode(file_bytes,1)

        result = NUCE(img)
        result = unsharp_masking(result,alpha)

        original_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        enhanced_rgb = cv.cvtColor(result,cv.COLOR_BGR2RGB)

        st.subheader("Before / After Comparison")

        image_comparison(
            img1=original_rgb,
            img2=enhanced_rgb,
            label1="Original",
            label2="Enhanced",
            width=700
        )

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(original_rgb)

        with col2:
            st.subheader("Enhanced Image")
            st.image(enhanced_rgb)

        st.subheader("Quality Metrics")

        psnr_value = calculate_psnr(img,result)
        ssim_value = calculate_ssim(img,result)

        m1,m2 = st.columns(2)

        m1.metric("PSNR",round(psnr_value,2))
        m2.metric("SSIM",round(ssim_value,3))

        _, buffer = cv.imencode(".png", result)

        st.download_button(
            label="⬇ Download Enhanced Image",
            data=buffer.tobytes(),
            file_name="enhanced_image.png",
            mime="image/png"
        )

# ---------------- VIDEO ----------------

elif mode=="Video Upload":

    uploaded_video = st.file_uploader("Upload Video",type=["mp4","avi","mov"])

    if uploaded_video:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        cap = cv.VideoCapture(temp_file.name)

        width = 480
        height = 320

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter("enhanced_video.mp4",fourcc,20.0,(width,height))

        col1,col2 = st.columns(2)

        original_frame = col1.empty()
        enhanced_frame = col2.empty()

        total_psnr = 0
        total_ssim = 0
        frame_count = 0

        while True:

            ret,frame = cap.read()

            if not ret:
                break

            frame = cv.resize(frame,(width,height))

            enhanced = NUCE(frame)
            enhanced = unsharp_masking(enhanced,alpha)

            out.write(enhanced)

            psnr_val = calculate_psnr(frame,enhanced)
            ssim_val = calculate_ssim(frame,enhanced)

            total_psnr += psnr_val
            total_ssim += ssim_val
            frame_count += 1

            frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            enhanced_rgb = cv.cvtColor(enhanced,cv.COLOR_BGR2RGB)

            original_frame.image(frame_rgb)
            enhanced_frame.image(enhanced_rgb)

        cap.release()
        out.release()

        if frame_count > 0:

            avg_psnr = total_psnr/frame_count
            avg_ssim = total_ssim/frame_count

            st.subheader("Video Quality Metrics")

            m1,m2 = st.columns(2)

            m1.metric("Average PSNR",round(avg_psnr,2))
            m2.metric("Average SSIM",round(avg_ssim,3))

        with open("enhanced_video.mp4","rb") as f:

            st.download_button(
                label="⬇ Download Enhanced Video",
                data=f,
                file_name="enhanced_video.mp4",
                mime="video/mp4"
            )

# ---------------- WEBCAM ----------------

elif mode=="Webcam Enhancement":

    run = st.button("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run:

        cap = cv.VideoCapture(0)

        while True:

            ret,frame = cap.read()

            if not ret:
                break

            frame = cv.resize(frame,(480,320))

            enhanced = NUCE(frame)
            enhanced = unsharp_masking(enhanced,alpha)

            combined = np.hstack((frame,enhanced))

            FRAME_WINDOW.image(
                cv.cvtColor(combined,cv.COLOR_BGR2RGB),
                channels="RGB"
            )

        cap.release()