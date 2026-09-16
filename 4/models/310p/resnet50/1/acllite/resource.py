"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2021-01-20 20:12:13
MODIFIED: 2021-02-03 14:04:45
"""
import acl
from acllite.logger import log_debug, log_info
import acllite.utils as utils

class AclLiteResource(object):
    """
    AclLiteResource
    """

    def __init__(self, device_id: int = 0):
        log_debug("acl resource init")
        self.device_id = device_id

        ret = acl.init()
        utils.check_ret("acl.init", ret)

        ret = acl.rt.set_device(self.device_id)
        utils.check_ret("acl.rt.set_device", ret)

        log_debug("acl resource init success")

    def __del__(self):
        log_debug("acl resource release")
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        log_info("acl resource released")
