import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import comfy.samplers
import comfy.sample
import comfy.utils
import os
import re
import cv2 # For Laplacian and Gradient calculations

class SamplerSchedulerMetricsTester:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "font_size_ratio": ("FLOAT", {"default": 0.025, "min": 0.01, "max": 0.1, "step": 0.005, "tooltip": "Font size relative to image height."}),
                "text_color": ("STRING", {"default": "white", "tooltip": "Color of the overlay text."}),
                "text_bg_color": ("STRING", {"default": "#000000A0", "tooltip": "Background color for the overlay text."}),
                "sampler_list_override": ("STRING", {"multiline": True, "default": "", "tooltip": "Comma or newline separated list of samplers. Leave empty for all."}),
                "scheduler_list_override": ("STRING", {"multiline": True, "default": "", "tooltip": "Comma or newline separated list of schedulers. Leave empty for all."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("Latents", "Overlaid Images", "Info")
    FUNCTION = "test_combinations"
    CATEGORY = "sampling/testing"

    def _pil_to_cv_gray(self, pil_image):
        if pil_image.mode == 'RGBA':
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
        elif pil_image.mode == 'RGB':
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif pil_image.mode == 'L':
            cv_image = np.array(pil_image)
        else:
            cv_image = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)

        if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        elif len(cv_image.shape) == 2:
            gray_image = cv_image
        else:
            raise ValueError(f"Unsupported image format for OpenCV conversion: mode {pil_image.mode}, shape {cv_image.shape}")
        return gray_image

    def _calculate_laplacian_variance(self, pil_image):
        if pil_image is None: return 0.0
        try:
            gray_image = self._pil_to_cv_gray(pil_image)
            return cv2.Laplacian(gray_image, cv2.CV_64F).var()
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error calculating Laplacian variance: {e}")
            return 0.0

    def _calculate_gradient_mean(self, pil_image):
        if pil_image is None: return 0.0
        try:
            gray_image = self._pil_to_cv_gray(pil_image)
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            return np.mean(gradient_magnitude)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error calculating Gradient Mean: {e}")
            return 0.0

    def _calculate_fft_sharpness_score(self, pil_image):
        if pil_image is None: return 0.0
        try:
            gray_image = np.array(pil_image.convert('L'))
            if gray_image.size == 0: return 0.0
            f = np.fft.fft2(gray_image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            log_magnitude_spectrum = np.log1p(magnitude_spectrum)
            return np.mean(log_magnitude_spectrum)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error calculating FFT sharpness: {e}")
            return 0.0

    def _parse_override_string(self, override_string, available_items_set, item_type_name="item"):
        if not override_string or override_string.strip() == "":
            return sorted(list(available_items_set))
        items_from_string = re.split(r'[,\n]+', override_string.strip())
        parsed_items = []
        seen_items = set()
        for item_str in items_from_string:
            cleaned_item_user_input = item_str.strip()
            if cleaned_item_user_input:
                matched_item = next((avail_item for avail_item in available_items_set if avail_item.lower() == cleaned_item_user_input.lower()), None)
                if matched_item:
                    if matched_item not in seen_items:
                        parsed_items.append(matched_item)
                        seen_items.add(matched_item)
                else:
                    print(f"[{self.__class__.__name__}] Warning: {item_type_name} '{cleaned_item_user_input}' from override list is not valid or known. Skipping.")
        if not parsed_items:
            print(f"[{self.__class__.__name__}] Warning: Override string for {item_type_name}s was provided but resulted in no valid items. Falling back to all available {item_type_name}s.")
            return sorted(list(available_items_set))
        print(f"[{self.__class__.__name__}] Using specified {item_type_name}s: {parsed_items}")
        return parsed_items

    def _get_pil_font(self, base_font_path="arial.ttf", fallback_font_path="DejaVuSans.ttf", size=12):
        try:
            return ImageFont.truetype(base_font_path, size)
        except IOError:
            try:
                if os.name == 'posix' and base_font_path.lower() == "arial.ttf":
                    if os.path.exists(fallback_font_path):
                         return ImageFont.truetype(fallback_font_path, size)
                return ImageFont.truetype(fallback_font_path, size)
            except IOError:
                print(f"[{self.__class__.__name__}] Warning: Preferred fonts ('{base_font_path}', '{fallback_font_path}') not found. Using PIL default.")
                return ImageFont.load_default()

    def test_combinations(self, model, vae, seed, steps, cfg, positive, negative, latent_image, denoise,
                          font_size_ratio=0.025, text_color="white", text_bg_color="#000000A0",
                          sampler_list_override="", scheduler_list_override=""):

        collected_latents_list = []
        collected_images_overlaid_list = []
        info_strings_list = []
        device = model.load_device
        initial_latent_tensor = latent_image["samples"]
        
        torch.manual_seed(seed)
        base_noise_cpu = torch.randn(initial_latent_tensor.shape,
                                     dtype=initial_latent_tensor.dtype,
                                     layout=initial_latent_tensor.layout,
                                     device="cpu")

        all_available_samplers = set(comfy.samplers.KSampler.SAMPLERS)
        all_available_schedulers = set(comfy.samplers.KSampler.SCHEDULERS)
        samplers_to_test = self._parse_override_string(sampler_list_override, all_available_samplers, "sampler")
        schedulers_to_test = self._parse_override_string(scheduler_list_override, all_available_schedulers, "scheduler")

        total_combinations = len(samplers_to_test) * len(schedulers_to_test)
        if total_combinations == 0:
            print(f"[{self.__class__.__name__}] Error: No valid sampler/scheduler combinations to test.")
            empty_latent_out = {"samples": torch.zeros_like(initial_latent_tensor)}
            img_h = initial_latent_tensor.shape[2] * 8; img_w = initial_latent_tensor.shape[3] * 8
            empty_image_out_tensor = torch.zeros((1, img_h, img_w, 3), dtype=torch.float32) 
            return (empty_latent_out, empty_image_out_tensor, "No combinations to test.")


        pbar = comfy.utils.ProgressBar(total_combinations)
        current_combination_idx_0_based = 0
        
        pil_font_loader = lambda s: self._get_pil_font(size=s)

        for sampler_name in samplers_to_test:
            for scheduler_name in schedulers_to_test:
                current_combination_idx_1_based = current_combination_idx_0_based + 1
                current_combination_info_str = f"{sampler_name} + {scheduler_name}"
                print(f"Testing combination ({current_combination_idx_1_based}/{total_combinations}): {current_combination_info_str}, Seed: {seed}")

                start_time = time.time()
                output_latent_samples_tensor = None
                processed_pil_image = None
                laplacian_score, gradient_score, fft_score = 0.0, 0.0, 0.0
                status_message = "OK"
                error_message_for_overlay = ""
                render_time_display_str = "N/A"

                try:
                    ksampler_helper = comfy.samplers.KSampler(model, steps, device,
                                                              sampler=sampler_name, scheduler=scheduler_name,
                                                              denoise=denoise, model_options=model.model_options)
                    current_sigmas = ksampler_helper.sigmas

                    if len(current_sigmas) == 0:
                        if denoise == 0.0:
                            output_latent_samples_tensor = initial_latent_tensor.clone().to(device)
                            render_time = 0.0; render_time_display_str = f"{render_time:.2f}s (Denoise 0)"
                        else:
                            raise ValueError("Sigma calculation resulted in empty sigmas with denoise > 0")
                    else:
                        current_iter_noise_gpu = base_noise_cpu.clone().to(device)
                        current_initial_latent_gpu = initial_latent_tensor.clone().to(device)
                        actual_sampler_kdiffusion_obj = comfy.samplers.sampler_object(sampler_name)

                        output_latent_samples_tensor = comfy.samplers.sample(
                            model, current_iter_noise_gpu, positive, negative, cfg, device,
                            actual_sampler_kdiffusion_obj, current_sigmas,
                            model_options=model.model_options, latent_image=current_initial_latent_gpu,
                            denoise_mask=latent_image.get("noise_mask"),
                            callback=None, disable_pbar=True, seed=seed
                        )
                        if isinstance(output_latent_samples_tensor, tuple):
                            output_latent_samples_tensor = output_latent_samples_tensor[0]
                    

                    pixel_image_tensor_bhwc = vae.decode(output_latent_samples_tensor.to(vae.device)).to(torch.float32)
                    # VAE output is BHWC: [Batch, Height, Width, Channels] as per logs: [1, 768, 768, 3]


                    end_time = time.time() 
                    render_time = end_time - start_time
                    if render_time_display_str == "N/A": 
                        render_time_display_str = f"{render_time:.2f}s"

                    # Since pixel_image_tensor_bhwc is BHWC, pixel_image_tensor_bhwc[0] is HWC
                    img_hwc_f32_cpu = pixel_image_tensor_bhwc[0].cpu()
                    
                    img_hwc_f32_cpu = torch.clamp(img_hwc_f32_cpu, 0.0, 1.0)
                    
                    # img_hwc_f32_cpu is already HWC, so no .permute(...) needed before .numpy()
                    img_hwc_u8_numpy = (img_hwc_f32_cpu.numpy() * 255).astype(np.uint8)
                                        
                    processed_pil_image = Image.fromarray(img_hwc_u8_numpy)

                    laplacian_score = self._calculate_laplacian_variance(processed_pil_image)
                    gradient_score = self._calculate_gradient_mean(processed_pil_image)
                    fft_score = self._calculate_fft_sharpness_score(processed_pil_image)

                    collected_latents_list.append({"samples": output_latent_samples_tensor.clone().cpu()})

                except Exception as e:
                    status_message = "ERROR"
                    error_message_for_overlay = str(e)[:100].replace("\n", " ")
                    print(f"ERROR during {current_combination_info_str} (Seed: {seed}): {e}") 
                    # import traceback # Uncomment for full traceback if needed
                    # traceback.print_exc()

                    time_taken_until_error = time.time() - start_time
                    render_time_display_str = f"({time_taken_until_error:.2f}s)"

                    img_w_pixels = initial_latent_tensor.shape[3] * 8; img_h_pixels = initial_latent_tensor.shape[2] * 8
                    processed_pil_image = Image.new("RGB", (img_w_pixels, img_h_pixels), "grey")
                    laplacian_score = self._calculate_laplacian_variance(processed_pil_image)
                    gradient_score = self._calculate_gradient_mean(processed_pil_image)
                    fft_score = self._calculate_fft_sharpness_score(processed_pil_image)
                    
                    if output_latent_samples_tensor is None: 
                        collected_latents_list.append({"samples": torch.zeros_like(initial_latent_tensor).cpu()})
                    else: 
                        collected_latents_list.append({"samples": output_latent_samples_tensor.clone().cpu()})


                image_numbering_text = f"Image: {current_combination_idx_1_based}/{total_combinations} (#{current_combination_idx_0_based})"
                
                text_header_overlay = f"{image_numbering_text}\n{current_combination_info_str}\nSeed: {seed}\nTime: {render_time_display_str}"
                if status_message == "ERROR":
                    text_header_overlay = f"ERROR: {image_numbering_text}\n{current_combination_info_str}\nSeed: {seed}\n{error_message_for_overlay}\nTime: {render_time_display_str}"

                text_metrics_overlay = (f"---\n"
                                        f"LapVar: {laplacian_score:.2f}\n"
                                        f"GradMean: {gradient_score:.2f}\n"
                                        f"FFTScore: {fft_score:.2f}")

                full_text_for_overlay = f"{text_header_overlay}\n{text_metrics_overlay}"
                
                summary_line_text = (f"({current_combination_idx_1_based}/{total_combinations}) "
                                     f"Status: {status_message} | "
                                     f"Combo: {current_combination_info_str} | Seed: {seed} | "
                                     f"Time: {render_time_display_str} | "
                                     f"LapVar: {laplacian_score:.2f} | "
                                     f"GradMean: {gradient_score:.2f} | "
                                     f"FFTScore: {fft_score:.2f}")
                if status_message == "ERROR":
                    summary_line_text += f" | Msg: {error_message_for_overlay}"

                if processed_pil_image:
                    actual_font_size = max(10, int(processed_pil_image.height * font_size_ratio))
                    current_pil_font = pil_font_loader(actual_font_size)

                    draw = ImageDraw.Draw(processed_pil_image, "RGBA")
                    text_x_main, text_y_main = 10, 10
                    
                    if text_bg_color and text_bg_color.lower() != "transparent":
                        try:
                            main_bbox = draw.textbbox((text_x_main, text_y_main), full_text_for_overlay, font=current_pil_font, spacing=4, align="left")
                            draw.rectangle([(main_bbox[0]-5, main_bbox[1]-5),(main_bbox[2]+5, main_bbox[3]+5)], fill=text_bg_color)
                        except Exception as e_draw_bg:
                             print(f"[{self.__class__.__name__}] Warning: Could not draw text background: {e_draw_bg}")

                    draw.text((text_x_main, text_y_main), full_text_for_overlay, font=current_pil_font, fill=text_color, spacing=4, align="left")
                    
                    img_overlaid_hwc_f32_numpy = np.array(processed_pil_image).astype(np.float32) / 255.0
                    img_overlaid_1hwc_tensor = torch.from_numpy(img_overlaid_hwc_f32_numpy).unsqueeze(0)
                    collected_images_overlaid_list.append(img_overlaid_1hwc_tensor.cpu())
                else:
                    h_err, w_err = initial_latent_tensor.shape[-2]*8, initial_latent_tensor.shape[-1]*8
                    collected_images_overlaid_list.append(torch.zeros((1, h_err, w_err, 3), dtype=torch.float32).cpu())


                info_strings_list.append(summary_line_text)
                pbar.update(1)
                current_combination_idx_0_based += 1

        if not collected_latents_list:
            empty_latent_out = {"samples": torch.zeros_like(initial_latent_tensor)}
            img_h = initial_latent_tensor.shape[2]*8; img_w = initial_latent_tensor.shape[3]*8
            empty_image_out_tensor = torch.zeros((1, img_h, img_w, 3), dtype=torch.float32)
            return (empty_latent_out, empty_image_out_tensor, "No combinations processed.")

        final_latents_batch_dict = {"samples": torch.cat([l['samples'] for l in collected_latents_list], dim=0)}
        final_images_overlaid_batch_tensor = torch.cat(collected_images_overlaid_list, dim=0) # This is (N, H, W, C)
        final_info_output_text = "\n".join(info_strings_list)

        return (final_latents_batch_dict, final_images_overlaid_batch_tensor, final_info_output_text)

NODE_CLASS_MAPPINGS = {
    "SamplerSchedulerMetricsTester": SamplerSchedulerMetricsTester
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerSchedulerMetricsTester": "Sampler Scheduler Metrics Tester"
}
