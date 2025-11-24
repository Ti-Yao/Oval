import sys
sys.path.append('utils')
from streamlit_utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -------------------------------
# Load Model
# -------------------------------
model_name = 'VESSEL-21'
model = tf.keras.models.load_model(
    f'models/{model_name}.h5',
    compile=False,
    custom_objects={'ResizeAndConcatenate': ResizeAndConcatenate}
)

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="Oval", page_icon="🟢", layout='wide')
st.subheader("Vessel Segmentation")

# -------------------------------
# Data and session state
# -------------------------------
data_path = '/workspaces/vessel/oval/data'
segmentation_list = pd.read_csv('oval_list.csv')

if "patient_index" not in st.session_state:
    st.session_state["patient_index"] = 0
if "point1" not in st.session_state:
    st.session_state["point1"] = None
if "point2" not in st.session_state:
    st.session_state["point2"] = None
if "segment" not in st.session_state:
    st.session_state["segment"] = False
if "seg_results" not in st.session_state:
    st.session_state["seg_results"] = {}
if "saved" not in st.session_state:
    st.session_state["saved"] = False
if "pick_second" not in st.session_state:
    st.session_state["pick_second"] = False
# -------------------------------
# Iterate patients
# -------------------------------
if st.session_state["patient_index"] < len(segmentation_list):
    row = segmentation_list.iloc[st.session_state["patient_index"]]
    mag_series_uid = row['mag_series_uid']
    phase_series_uid = row['phase_series_uid']
    vessel = row['vessel']
    redcap_flow = row['redcap_flow']

    p = OvalPipeline(data_path, mag_series_uid, phase_series_uid)
    image, venc, rr, description, patient, study_date = p.image, p.venc, p.rr, p.description, p.patient, p.studydate

    st.markdown(f"**Mag Series UID:** {mag_series_uid}  |  **Phase Series UID:** {phase_series_uid}")
    st.markdown(f"**Patient:** {patient} | **Study Date:** {study_date}  |  **Description:** {description}")
    st.markdown(f"**Vessel:** {vessel.upper()}  |  **RR:** {rr}  |  **VENC:** {venc}")

    col1, col2, col3, col4, col5 = st.columns(5)
    mag_image, phase_image = image[..., 0], image[...,1]

    slice_2d = mag_image[..., 0]
    print(mag_image.shape)
    slice_2d = (255 * (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d) + 1e-8)).astype(np.uint8)
    display_width = slice_2d.shape[0]
    os.makedirs('results/segs', exist_ok=True)
    os.makedirs('results/gifs', exist_ok=True)
    os.makedirs('results/curve_plot', exist_ok=True)

    # -------------------------------
    # Column 1: Pick points
    # -------------------------------
    with col1:
        
        st.caption("Pick first point:")
        img1 = Image.fromarray(slice_2d).convert("RGB")
        if st.session_state["point1"] is not None:
            draw1 = ImageDraw.Draw(img1)
            draw1.ellipse(get_ellipse_coords(st.session_state["point1"]), fill=colorlist[vessel])

        def add_point1():
            raw_value = st.session_state.get("coord1")
            if raw_value:
                st.session_state["point1"] = (raw_value["x"], raw_value["y"])
                st.session_state["segment"] = False  # mark segmentation outdated

        coords1 = streamlit_image_coordinates(img1, key='coord1', cursor='crosshair', on_click=add_point1, width = image.shape[0])

        pick_second = st.toggle(
            "Pick a second point?",
            value=False,  # always start unticked
            key=f"pick_second_{st.session_state['patient_index']}"
        )

        coords2 = None
        if pick_second:
            st.caption("Pick second point:")
            img2 = Image.fromarray(slice_2d).convert("RGB")
            if st.session_state["point2"] is not None:
                draw2 = ImageDraw.Draw(img2)
                draw2.ellipse(get_ellipse_coords(st.session_state["point2"]), fill=colorlist[vessel])

            def add_point2():
                raw_value = st.session_state.get("coord2")
                if raw_value:
                    st.session_state["point2"] = (raw_value["x"], raw_value["y"])
                    st.session_state["segment"] = False

            coords2 = streamlit_image_coordinates(img2, key='coord2', cursor='crosshair', on_click=add_point2, width = image.shape[0])

    # -------------------------------
    # Column 2: Cropped images
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

        if coords1 is not None and st.button("Segment Images!", type="primary", key="segment_button", width=display_width):
            st.session_state.segment = True

    # -------------------------------
    # Column 3–5: Segmentation, GIF caching, flow, save
    # -------------------------------
    if st.session_state.segment and coords1 is not None:
        seg_key = mag_series_uid
        coords_hash = (st.session_state["point1"], st.session_state["point2"])

        # Check session cache
        if seg_key in st.session_state["seg_results"]:
            stored = st.session_state["seg_results"][seg_key]
            if stored.get("coords") == coords_hash:
                pred_masks = stored["pred_masks"]
                gif_paths = stored["gif_paths"]
            else:
                pred_masks, gif_paths = [], []
        else:
            pred_masks, gif_paths = [], []


        # -------------------------------
        # Display GIFs
        # -------------------------------
        with col3:
            # Run segmentation if needed
            if not pred_masks:
                with st.spinner("Segmenting Images..."):
                    plot_image1, plot_mask1, pred_mask1 = segment_image(image, venc, model, x_min1, x_max1, y_min1, y_max1)
                    gif_path1 = f"results/temp_point1.gif"
                    make_video(plot_image1, plot_mask1, vessel, gif_path1)
                    pred_masks.append(pred_mask1)
                    gif_paths.append(gif_path1)

                    if coords2 is not None:
                        plot_image2, plot_mask2, pred_mask2 = segment_image(image, venc, model, x_min2, x_max2, y_min2, y_max2)
                        gif_path2 = f"results/temp_point2.gif"
                        make_video(plot_image2, plot_mask2, vessel, gif_path2)
                        pred_masks.append(pred_mask2)
                        gif_paths.append(gif_path2)

                # Cache results
                st.session_state["seg_results"][seg_key] = {
                    "pred_masks": pred_masks,
                    "gif_paths": gif_paths,
                    "coords": coords_hash
                }

            for idx, file_path in enumerate(gif_paths, start=1):
                st.caption(f"Mask {idx}:")
                display_gif(file_path, width=display_width)

        # -------------------------------
        # Compute flows and display
        # -------------------------------
        with col4:
            st.caption("Segmented Magnitude GIF:")
            mask = np.max(pred_masks, axis=0)

            gif_cache_key_mag = f"{mag_series_uid}_gif"
            gif_cache_key_phase = f"{mag_series_uid}_phase_gif"

            # Magnitude GIF cache
            if gif_cache_key_mag in st.session_state and st.session_state[gif_cache_key_mag]["coords"] == coords_hash:
                gif_path = st.session_state[gif_cache_key_mag]["path"]
            else:
                gif_path = f'results/segs/{mag_series_uid}.gif'
                make_video(mag_image, mask, vessel, gif_path)
                st.session_state[gif_cache_key_mag] = {"path": gif_path, "coords": coords_hash}

            display_gif(gif_path, width=display_width)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Corresponding Phase GIF:")

            # Phase GIF cache
            if gif_cache_key_phase in st.session_state and st.session_state[gif_cache_key_phase]["coords"] == coords_hash:
                gif_phase_path = st.session_state[gif_cache_key_phase]["path"]
            else:
                gif_phase_path = f'results/segs/{mag_series_uid}_phase.gif'
                make_video(imaginary_image, mask, vessel, gif_phase_path, alpha=0)
                st.session_state[gif_cache_key_phase] = {"path": gif_phase_path, "coords": coords_hash}

            display_gif(gif_phase_path, width=display_width)


        # -------------------------------
        # Save flows and advance
        # -------------------------------
        with col5:
            flow_curve, flow, total_volume, forward_volume, backward_volume = calculate_flow(phase_image, mask, rr, vessel)

            st.caption("Volume Curve:")
            fig, ax = plt.subplots(1, 1, figsize=(4, 3.8))
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
            plt.savefig(f'results/curve_plot/{mag_series_uid}.png', bbox_inches = 'tight')

            st.pyplot(fig, width=int(display_width * 1.4))
            plt.close()

            combine_gif_png(gif_path = f'results/segs/{mag_series_uid}.gif', 
                            png_path = f'results/curve_plot/{mag_series_uid}.png', 
                            output_path = f'results/gifs/{mag_series_uid}.gif')

            difference = (redcap_flow - flow) * 100 / redcap_flow
            diff_color = "#11AC35" if difference < 10 else "#d9534f"
            st.caption("Flows:")
            st.markdown(f"""
                <table style="
                    width:{int(display_width * 0.9)}px;
                    border-collapse:collapse;
                    background-color:#F7F7F7;
                    font-size:16px;
                    margin:auto;
                ">
                    <tr>
                        <td style="padding:4px 8px; font-weight:600; color:#555;">Redcap (L/min)</td>
                        <td style="padding:4px 8px; font-weight:600; text-align:right; color:#555;">{redcap_flow:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px; font-weight:600; color:#555;">DL (L/min)</td>
                        <td style="padding:4px 8px; font-weight:600;  text-align:right; color:#555;">{flow:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 8px; font-weight:600; color:{diff_color};">Difference (%)</td>
                        <td style="padding:4px 8px; font-weight:600; text-align:right; color:{diff_color};">{difference:.1f}</td>
                    </tr>
                </table>
            """, unsafe_allow_html=True)

            st.write('\n')
            # Save button
            if st.button("Save Flow  💾", type="primary", key="save_flows", width=display_width):
                predicted_df = pd.DataFrame({
                    'patient': [patient],
                    'vessel': [vessel],
                    'mag_series_uid': [mag_series_uid],
                    'phase_series_uid': [phase_series_uid],
                    'description': [description],
                    'flow': [flow],
                    'total_volume': [total_volume],
                    'forward_volume': [forward_volume],
                    'backward_volume': [backward_volume],
                    'rr': [rr],
                    'venc': [venc],
                    'mask':[mask],
                    'flow_curve':[flow_curve],
                    'point1_x': [st.session_state["point1"][0]],
                    'point1_y': [st.session_state["point1"][1]],
                    'point2_x': [st.session_state.get("point2")[0] if st.session_state.get("point2") is not None else None],
                    'point2_y': [st.session_state.get("point2")[1] if st.session_state.get("point2") is not None else None],
                })

                os.makedirs('results/flow_curves', exist_ok=True)
                os.makedirs('results/masks', exist_ok=True)
                os.makedirs('results/dfs', exist_ok=True)
                predicted_df.to_csv(f'results/dfs/{mag_series_uid}.csv')
                np.save(f'results/flow_curves/{mag_series_uid}.npy', flow_curve)
                nii_mask = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
                nib.save(nii_mask, f'results/masks/{mag_series_uid}.nii.gz')
                st.session_state.saved = True
                st.success("Flows saved successfully!   ✅", width=display_width)

            # Next scan button (enabled only after saving)
            if st.session_state.get("saved", False):
                if st.button("Next Scan  ➡️", type = 'primary', width=display_width):
                    st.session_state["patient_index"] += 1
                    st.session_state["segment"] = False
                    st.session_state["point1"] = None
                    st.session_state["point2"] = None
                    if "coord1" in st.session_state: del st.session_state["coord1"]
                    if "coord2" in st.session_state: del st.session_state["coord2"]
                    st.session_state["pick_second"] = False
                    st.session_state['saved'] = False
                    st.rerun()
else:
    st.success("All patients processed!   🎉🎉🎉")
