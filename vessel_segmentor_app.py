from streamlit_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -------------------------------
# THINGS TO CHANGE GABE
# -------------------------------
data_path = '/workspaces/vessel/streamlit/data'
segmentation_list = pd.read_csv('vessel_segmentor_list.csv')

mag_series_uid = '1.3.6.1.4.1.53684.1.1.3.1887816670.2028.1636387520.819542'
phase_series_uid = '1.3.6.1.4.1.53684.1.1.3.1887816670.2028.1636387520.819542'
image, venc, rr, description, patient, study_date =  get_image(data_path, mag_series_uid, phase_series_uid)

vessel = 'ao' # GABE
redcap_flow = 2.9 # GABE

# -------------------------------
# Load Model
# -------------------------------
model_name = 'VESSEL-19'
model = tf.keras.models.load_model(f'models/{model_name}.h5', compile=False, custom_objects={'ResizeAndConcatenate': ResizeAndConcatenate})

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="Oval", page_icon="🟢", layout='wide')
st.subheader("Vessel Segmentation")
st.markdown(f"**Mag Series UID:** {mag_series_uid}  |  **Phase Series UID:** {phase_series_uid}")
st.markdown(f"**Patient:** {patient} | **Study Date:** {study_date}  |  **Description:** {description}")
st.markdown(f"**Vessel:** {vessel.upper()}  |  **RR:** {rr}  |  **VENC:** {venc}")

col1, col2, col3, col4, col5 = st.columns(5)

# -------------------------------
# Session state defaults
# -------------------------------
if "point1" not in st.session_state:
    st.session_state["point1"] = None
if "point2" not in st.session_state:
    st.session_state["point2"] = None
if "segment" not in st.session_state:
    st.session_state["segment"] = False
if "saved" not in st.session_state:
    st.session_state["saved"] = False


# -------------------------------
# Column 1: Pick points
# -------------------------------
with col1:
    mag_image = image[..., 0]
    slice_2d = mag_image[..., 0]
    slice_2d = (255 * (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d) + 1e-8)).astype(np.uint8)
    display_width = slice_2d.shape[0]

    os.makedirs(f'results/segs', exist_ok=True)

    # First point
    st.caption("Pick first point:")
    img1 = Image.fromarray(slice_2d).convert("RGB")
    if st.session_state["point1"] is not None:
        draw1 = ImageDraw.Draw(img1)
        draw1.ellipse(get_ellipse_coords(st.session_state["point1"]), fill=colorlist[vessel])

    def add_point1():
        raw_value = st.session_state["coord1"]
        st.session_state["point1"] = (raw_value["x"], raw_value["y"])

    coords1 = streamlit_image_coordinates(img1, key="coord1", cursor='crosshair', on_click=add_point1)

    # Optional second point
    coords2 = None
    if st.checkbox("Pick a second point?", value=False):
        st.caption("Pick second point:")
        img2 = Image.fromarray(slice_2d).convert("RGB")
        if st.session_state["point2"] is not None:
            draw2 = ImageDraw.Draw(img2)
            draw2.ellipse(get_ellipse_coords(st.session_state["point2"]), fill=colorlist[vessel])

        def add_point2():
            raw_value = st.session_state["coord2"]
            st.session_state["point2"] = (raw_value["x"], raw_value["y"])

        coords2 = streamlit_image_coordinates(img2, key="coord2", cursor='crosshair', on_click=add_point2)

# -------------------------------
# Column 2: Cropped images and segmentation trigger
# -------------------------------
with col2:
    if coords1 is not None:
        st.caption("Cropped Image 1:")
        x_min1, x_max1, y_min1, y_max1 = get_crop_coords(coords1)
        st.image(slice_2d[y_min1:y_max1, x_min1:x_max1], width=display_width)

    if coords2 is not None:
        st.caption("Cropped Image 2:")
        x_min2, x_max2, y_min2, y_max2 = get_crop_coords(coords2)
        st.image(slice_2d[y_min2:y_max2, x_min2:x_max2], width=display_width)

    if coords1 is not None and st.button("Segment Images!", type="primary", width=display_width):
        st.session_state.segment = True

# -------------------------------
# Column 3-5: Run segmentation only if triggered
# -------------------------------
    if st.session_state.segment and coords1 is not None:
        pred_masks = []
        with st.spinner("Segmenting Images..."):
            gif_paths = []

            # Point 1 segmentation
            if coords1 is not None:
                plot_image1, plot_mask1, pred_mask1 = segment_image(image, venc, model, x_min1, x_max1, y_min1, y_max1)
                gif_path1 = f"results/temp_point1.gif"
                make_video(plot_image1, plot_mask1, vessel, gif_path1)
                gif_paths.append(gif_path1)
                pred_masks.append(pred_mask1)

            # Point 2 segmentation
            if coords2 is not None:
                plot_image2, plot_mask2, pred_mask2 = segment_image(image, venc, model, x_min2, x_max2, y_min2, y_max2)
                gif_path1 = f"results/temp_point2.gif"
                make_video(plot_image2, plot_mask2, vessel, gif_path1)
                gif_paths.append(gif_path1)
                pred_masks.append(pred_mask2)

    # Display GIFs in column 3
    with col3:
        if st.session_state.segment:
            for idx, file_path in enumerate(gif_paths, start=1):
                st.caption(f"Mask {idx}:")
                display_gif(file_path, width=display_width)
                st.write("\n\n")

    def calculate_flow(phase_image, mask):
        flow_curve = calculate_curve(mask, phase_image, vessel)
        flow_curve = interpolate_curve(flow_curve, rr)
        flow = np.mean(flow_curve) * 0.06
        total_volume = np.sum(flow_curve)/1000
        forward_volume = np.sum(flow_curve[flow_curve>0])/1000
        backward_volume = abs(np.sum(flow_curve[flow_curve<0])/1000)
        return flow_curve, flow, total_volume, forward_volume, backward_volume

    # Combined mask and flow computation in column 4
    with col4:
        if st.session_state.segment:
            st.caption(f"Segmented GIF:")
            mag_image, phase_image = image[...,0], image[...,1]
            mask = np.max(pred_masks, axis=0)
            gif_path = f'results/segs/{mag_series_uid}.gif'
            make_video(mag_image, mask, vessel, gif_path)
            display_gif(gif_path, width=display_width)

            flow_curve, flow, total_volume, forward_volume, backward_volume = calculate_flow(phase_image, mask)

            
            st.caption(f"Volume Curve:")
            # plot curve
            fig, ax = plt.subplots(1,1, figsize=(4,3.8))
            ax.plot(flow_curve, linewidth=4, c=colormaps[vessel].colors[0])
            fontsize = 12
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
            ax.set_ylabel('Flow (mL/s)', fontsize=fontsize, rotation=270, labelpad=15, va='center')
            ax.set_xlabel('Time (ms)', fontsize=fontsize)
            ax.set_title(f'{vessel.upper()} | {flow:.2f} L/min | {total_volume:.1f} mL | (V{int(venc)})', fontsize=15)
            ax.set_xlim(0, len(flow_curve))
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            st.pyplot(fig, width=int(display_width * 1.2))
            plt.close()

    # Display flow comparison and save button in column 5
    with col5:
        if st.session_state.segment:
            difference = (redcap_flow - flow) * 100 / redcap_flow
            difference_color = "#258d3d" if difference < 10 else "#d9534f"

            st.caption(f"Flows:")
            st.markdown(f"""
                <div style="
                    background-color:#f8f9fa;
                    border:1px solid #ddd;
                    border-radius:10px;
                    padding:12px 12px;
                    box-shadow:0 1px 1px rgba(0,0,0,0.1);
                    width:{display_width}px;
                ">
                    <table style="width:100%; border-collapse:collapse;">
                        <tr>
                            <td style="color:#555; font-weight:600;">Redcap (L/min) </td>
                            <td style="text-align:right; color:#555 font-weight:600;">{redcap_flow:.2f}</td>
                        </tr>
                        <tr>
                            <td style="color:#555; font-weight:600;">DL (L/min)</td>
                            <td style="text-align:right; color:#555 font-weight:600;">{flow:.2f}</td>
                        </tr>
                        <tr>
                            <td style="color:{difference_color}; font-weight:600;">Difference (%)</td>
                            <td style="text-align:right; color:{difference_color}; font-weight:600;">{difference:.1f}</td>
                        </tr>
                    </table>
                </div>
            """, unsafe_allow_html=True)

            # save. THIS NEEDS TO BE EDITED.
            if st.button("Save Flows?", type="primary", width=display_width):
                predicted_df = pd.DataFrame({
                    'patient': [patient],
                    'vessel': [vessel],
                    'mask': [mask],
                    'mag_series_uid': [mag_series_uid],
                    'phase_series_uid': [phase_series_uid],
                    'description': [description],
                    'flow': [flow],
                    'total_volume': [total_volume],
                    'forward_volume': [forward_volume],
                    'backward_volume': [backward_volume],
                    'flow_curve': [flow_curve],
                    'rr': [rr],
                    'venc': [venc]
                })
                os.makedirs(f'results/flow_curves', exist_ok=True)
                os.makedirs('results/masks', exist_ok=True)
                predicted_df.to_csv(f'results/{mag_series_uid}.csv')
                np.save(f'results/flow_curves/{mag_series_uid}.npy', flow_curve)
                nii_mask = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
                nib.save(nii_mask, f'results/masks/{mag_series_uid}.nii.gz')
                st.session_state.saved = True
                st.success("Flows saved successfully!", width=display_width)
