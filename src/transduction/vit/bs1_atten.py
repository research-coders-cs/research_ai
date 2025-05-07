import torch
import numpy as np
from torchvision.transforms import ToPILImage
transform_to_pil = ToPILImage()
from ..plot_if import get_plt, plt_imshow
plt = get_plt()

from ..vit_finetune.attention import get_mask, generate_attention_heatmap

class Bs1Atten:

    @staticmethod
    def save(pixels, labels, logits, attentions, i_batch):
        pred = logits.argmax(-1)[0].item()
        true = labels[0].item()
        print(f"Predicted: {pred}, True: {true}")

        print(f'saving bs1_attn_{i_batch}.pt')  # ~7.3MB
        torch.save({'pixels': pixels,
                    'true': true,
                    'pred': pred,
                    'attentions': attentions}, f'bs1_attn_{i_batch}.pt')

    @staticmethod
    def load(pt_file_path):
        debug_attn = torch.load(pt_file_path)  # ~7.3MB
        pixels = debug_attn['pixels']
        true = debug_attn['true']
        pred = debug_attn['pred']
        attentions = debug_attn['attentions']

        print(f"Predicted: {pred}, True: {true}")
        print(f"Number of attention layers: {len(attentions)}")

        return pixels, attentions


    @staticmethod
    def process(pixels, attentions, i_batch):
        for i, attn in enumerate(attentions):
            print(f"Layer {i+1} attention shape: {attn.shape}")  # torch.Size([1, 12, 197, 197])
            # 1 samples, 12 heads (from ViT-Base), 197 tokens (1 CLS + 196 patches).
            """
            The 12 comes from the architecture of the pretrained "google/vit-base-patch16-224" model, which your CustomViT inherits via self.encoder = pretrained_vit.encoder:
            - ViT-Base Specification:
              -- Hidden size: 768.
              -- Number of layers: 12 (using 4 via num_hidden_layers=4).
              -- Number of attention heads: 12 (fixed in the pretrained model).
            - Multi-Head Attention: Each transformer layer in ViT-Base uses multi-head self-attention with 12 heads. Each head processes a portion of the hidden size (768 / 12 = 64 dimensions per head), allowing the model to attend to different aspects of the input simultaneously.
            """
            """
            # Why Not Customizable Here?
            The 12 is hardcoded in the pretrained "google/vit-base-patch16-224" weights. Since you’re leveraging pretrained benefits, the number of heads is fixed to match the loaded weights. If you wanted a different number (e.g., 8), you’d need to:
            - Redefine the attention layers from scratch (losing pretrained weights).
            - Or modify the pretrained layers post-loading (complex and risks breaking compatibility).
            For your MNIST task with pretrained weights, sticking with 12 is optimal and aligns with the original ViT-Base design.

            # What It Means Practically
            - 12 Heads: Each head learns to focus on different patterns in the 197 tokens (CLS + 196 patches). For MNIST, this might mean different heads attend to different parts of the digit (e.g., edges, curves).
            - Debugging: When you inspect attn[0, 0, 0, :5], you’re looking at the first head’s attention from the CLS token to the first 5 patches. The 12 heads give you 12 such perspectives per layer.
            """

            print(f"CLS attention to first 5 patches: {attn[0, 0, 0, :5]}")

        att_mat = torch.cat(attentions).cpu()
        #print(f'@@ att_mat.shape: {att_mat.shape}')  # torch.Size([4, 12, 197, 197]); num_hidden_layers, num_heads, seq_len, seq_len

        #print('@@ (bs1) pixels.shape:', pixels.shape)  # torch.Size([1, 1, 224, 224])
        input = pixels.cpu()[0, 0, :, :]  # torch.Size([224, 224])

        Bs1Atten.compute_heatmap(input, att_mat, i_batch)  # averaged across all heads

        for i_head in range(att_mat.shape[1]):
            Bs1Atten.compute_heatmap(input, att_mat, i_batch, i_head)


    @staticmethod
    def compute_heatmap(input, att_mat, i_batch, i_head=None, im_orig=None):
        if i_head is None:
            # Average the attention weights across all heads.
            head_att_mat = torch.mean(att_mat, dim=1)  # torch.Size([4, 197, 197]); num_hidden_layers, seq_len, seq_len
            tag = 'ave'
        else:
            head_att_mat = att_mat[:, i_head, :, :]  # Shape: [4, 197, 197], {i_head}_head_attention
            tag = f'{i_head}'

        im_mask, _, _ = get_mask(transform_to_pil(input), head_att_mat)
        #plt_imshow(plt, im_mask)  # debug (im_mask.shape: (224, 224))

        if im_orig is None:
            # Restoring from `pixels`, could be visually affected by proprocessing
            im_orig = np.stack((input.numpy(),) * 3, axis=-1)  # (224, 224, 3)
        #plt_imshow(plt, im_orig)  # debug

        im_heatmap = transform_to_pil(generate_attention_heatmap(im_orig, im_mask))
        fname = f'bs1_attn_heatmap_batch_{i_batch}_head_{tag}.png'
        #plt_imshow(plt, im_heatmap)

        print(f'saving {fname}')
        im_heatmap.save(fname)

        return im_heatmap, im_mask