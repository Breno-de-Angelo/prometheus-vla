import sys
print(sys.path)
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    print("Import successful")
    ChannelFactoryInitialize(0)
    print("Init successful")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
