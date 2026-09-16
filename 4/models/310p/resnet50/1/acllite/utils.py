import numpy as np
import acl
import acllite.constants as const

def check_ret(message, ret_int):
    """Check int value is 0 or not
    Args:
        message: output log str
        ret_int: check value that type is int
    """
    if ret_int != 0:
        raise RuntimeError(f"{message} failed with error code {ret_int}")

def check_none(message, ret_none):
    """Check object is None or not
    Args:
        message: output log str
        ret_none: check object
    """
    if ret_none is None:
        raise Exception("{} failed"
                        .format(message))

def copy_data_to_dvpp(data, size, run_mode):
    """Copy data to dvpp
    Args:
        data: data that to be copyed
        data_size: data size
        run_mode: device run mode
    Returns:
        None: copy failed
        others: data which copy from host_data
    """
    policy = const.ACL_MEMCPY_HOST_TO_DEVICE
    if run_mode == const.ACL_DEVICE:
        policy = const.ACL_MEMCPY_DEVICE_TO_DEVICE

    dvpp_buf, ret = acl.media.dvpp_malloc(size)
    check_ret("acl.rt.malloc_host", ret)

    ret = acl.rt.memcpy(dvpp_buf, size, data, size, policy)
    check_ret("acl.rt.memcpy", ret)

    return dvpp_buf

def align_up(value, align):
    """Align up int value
    Args:
        value:input data
        align: align data
    Return:
        aligned data
    """
    return int(int((value + align - 1) / align) * align)

def align_up16(value):
    """Align up data with 16
    Args:
        value:input data
    Returns:
        16 aligned data
    """
    return align_up(value, 16)

def align_up64(value):
    """Align up data with 128
    Args:
        value:input data
    Returns:
        128 aligned data
    """
    return align_up(value, 64)

def align_up128(value):
    """Align up data with 128
    Args:
        value:input data
    Returns:
        128 aligned data
    """
    return align_up(value, 128)

def align_up2(value):
    """Align up data with 2
    Args:
        value:input data
    Returns:
        2 aligned data
    """
    return align_up(value, 2)

def yuv420sp_size(width, height):
    """Calculate yuv420sp image size
    Args:
        width: image width
        height: image height
    Returns:
        image data size
    """
    return int(width * height * 3 / 2)

def rgbu8_size(width, height):
    """Calculate rgb 24bit image size
    Args:
        width: image width
        height: image height
    Returns:
        rgb 24bit image data size
    """
    return int(width * height * 3)

def to_numpy_type(data_type: int) -> type:
    """Convert acl data type to numpy data type
    Args:
        data_type: acl data type
    Returns:
        numpy data type
    """
    np_types = {
        const.ACL_FLOAT16: np.float16,
        const.ACL_FLOAT:  np.float32,
        const.ACL_DOUBLE: np.float64,
        const.ACL_INT8: np.int8,
        const.ACL_UINT8: np.uint8,
        const.ACL_INT16: np.int16,
        const.ACL_UINT16: np.uint16,
        const.ACL_INT32: np.int32,
        const.ACL_UINT32: np.uint32,
        const.ACL_INT64: np.int64,
        const.ACL_UINT64: np.uint64,
        const.ACL_BOOL: np.bool_,
    }
    if data_type in np_types:
        return np_types[data_type]
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
