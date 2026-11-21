import random
import cv2

def make_patch(pre, post, mask, patch_size=256, out_size=512):

    # obtain the position of the patch
    height,width = mask.shape
    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)

    # crop the patch from the original images
    pre = pre[top:top + patch_size, left:left + patch_size]
    post = post[top:top + patch_size, left:left + patch_size]
    mask = mask[top:top + patch_size, left:left + patch_size]

    # resize the patch (512x512)
    pre = cv2.resize(pre, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    post = cv2.resize(post, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    return pre, post, mask