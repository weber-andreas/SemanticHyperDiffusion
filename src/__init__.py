import os
import sys

# Add external directory to sys.path so that modules inside external can import each other
# e.g. Pointnet_Pointnet2_pytorch imports itself
sys.path.append(os.path.join(os.path.dirname(__file__), "external"))
