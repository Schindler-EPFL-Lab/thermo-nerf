"""
Train a radiance field with nerfstudio.
"""
import sys

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
sys.path.append("./nerfstudio")

from nerfstudio.scripts.train import entrypoint  # noqa: E402

if __name__ == "__main__":
    entrypoint()
