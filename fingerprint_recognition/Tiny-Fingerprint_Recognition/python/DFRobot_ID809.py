"""
DFRobot_ID809 - Python Library for SEN0348 Capacitive Fingerprint Sensor
Ported from Arduino C++ library for use with Jetson Nano (UART)

Hardware Connection (Jetson Nano B01 <-> SEN0348):
    Jetson Pin 1  (3.3V)  -> SEN0348 VCC (nếu module hỗ trợ 3.3V, nếu không dùng 5V từ Pin 2/4)
    Jetson Pin 6  (GND)   -> SEN0348 GND
    Jetson Pin 8  (TXD)   -> SEN0348 RX
    Jetson Pin 10 (RXD)   -> SEN0348 TX

UART: /dev/ttyTHS1 (UART2 trên Jetson Nano B01), baudrate mặc định 115200
"""

import serial
import struct
import time
from enum import IntEnum


# ============== Constants ==============
CMD_PREFIX_CODE      = 0xAA55
RCM_PREFIX_CODE      = 0x55AA
CMD_DATA_PREFIX_CODE = 0xA55A
RCM_DATA_PREFIX_CODE = 0x5AA5

CMD_TYPE  = 0xF0
RCM_TYPE  = 0xF0
DATA_TYPE = 0x0F

# Commands
CMD_TEST_CONNECTION      = 0x0001
CMD_SET_PARAM            = 0x0002
CMD_GET_PARAM            = 0x0003
CMD_DEVICE_INFO          = 0x0004
CMD_SET_MODULE_SN        = 0x0008
CMD_GET_MODULE_SN        = 0x0009
CMD_ENTER_STANDBY_STATE  = 0x000C
CMD_GET_IMAGE            = 0x0020
CMD_FINGER_DETECT        = 0x0021
CMD_UP_IMAGE_CODE        = 0x0022
CMD_DOWN_IMAGE           = 0x0023
CMD_SLED_CTRL            = 0x0024
CMD_STORE_CHAR           = 0x0040
CMD_LOAD_CHAR            = 0x0041
CMD_UP_CHAR              = 0x0042
CMD_DOWN_CHAR            = 0x0043
CMD_DEL_CHAR             = 0x0044
CMD_GET_EMPTY_ID         = 0x0045
CMD_GET_STATUS           = 0x0046
CMD_GET_BROKEN_ID        = 0x0047
CMD_GET_ENROLL_COUNT     = 0x0048
CMD_GET_ENROLLED_ID_LIST = 0x0049
CMD_GENERATE             = 0x0060
CMD_MERGE                = 0x0061
CMD_MATCH                = 0x0062
CMD_SEARCH               = 0x0063
CMD_VERIFY               = 0x0064

ERR_SUCCESS = 0x00
ERR_ID809   = 0xFF

MODULE_SN_SIZE = 16
DELALL = 0xFF


class LEDMode(IntEnum):
    BREATHING  = 1
    FAST_BLINK = 2
    KEEPS_ON   = 3
    NORMAL_CLOSE = 4
    FADE_IN    = 5
    FADE_OUT   = 6
    SLOW_BLINK = 7


class LEDColor(IntEnum):
    GREEN   = 1
    RED     = 2
    YELLOW  = 3
    BLUE    = 4
    CYAN    = 5
    MAGENTA = 6
    WHITE   = 7


class DeviceBaudrate(IntEnum):
    BPS_9600   = 1
    BPS_19200  = 2
    BPS_38400  = 3
    BPS_57600  = 4
    BPS_115200 = 5


class Error(IntEnum):
    SUCCESS           = 0x00
    FAIL              = 0x01
    VERIFY            = 0x10
    IDENTIFY          = 0x11
    TMPL_EMPTY        = 0x12
    TMPL_NOT_EMPTY    = 0x13
    ALL_TMPL_EMPTY    = 0x14
    EMPTY_ID_NOEXIST  = 0x15
    BROKEN_ID_NOEXIST = 0x16
    INVALID_TMPL_DATA = 0x17
    DUPLICATION_ID    = 0x18
    BAD_QUALITY       = 0x19
    MERGE_FAIL        = 0x1A
    NOT_AUTHORIZED    = 0x1B
    MEMORY            = 0x1C
    INVALID_TMPL_NO   = 0x1D
    INVALID_PARAM     = 0x22
    TIMEOUT           = 0x23
    GEN_COUNT         = 0x25
    INVALID_BUFFER_ID = 0x26
    FP_NOT_DETECTED   = 0x28
    FP_CANCEL         = 0x41
    RECV_LENGTH       = 0x42
    RECV_CKS          = 0x43
    GATHER_OUT        = 0x45
    RECV_TIMEOUT      = 0x46


ERROR_DESCRIPTIONS = {
    Error.SUCCESS:           "Command processed successfully",
    Error.FAIL:              "Command processing failed",
    Error.VERIFY:            "1:1 comparison failed",
    Error.IDENTIFY:          "Comparison with all fingerprints failed",
    Error.TMPL_EMPTY:        "No fingerprint in designated ID",
    Error.TMPL_NOT_EMPTY:    "Designated ID has fingerprint",
    Error.ALL_TMPL_EMPTY:    "Module unregistered fingerprint",
    Error.EMPTY_ID_NOEXIST:  "No registerable ID here",
    Error.BROKEN_ID_NOEXIST: "No broken fingerprint",
    Error.INVALID_TMPL_DATA: "Invalid designated fingerprint data",
    Error.DUPLICATION_ID:    "The fingerprint has been registered",
    Error.BAD_QUALITY:       "Poor quality fingerprint image",
    Error.MERGE_FAIL:        "Fingerprint synthesis failed",
    Error.NOT_AUTHORIZED:    "Communication password not authorized",
    Error.MEMORY:            "External Flash burning error",
    Error.INVALID_TMPL_NO:   "Invalid designated ID",
    Error.INVALID_PARAM:     "Incorrect parameter",
    Error.TIMEOUT:           "Acquisition timeout",
    Error.GEN_COUNT:         "Invalid number of fingerprint synthesis",
    Error.INVALID_BUFFER_ID: "Incorrect Buffer ID value",
    Error.FP_NOT_DETECTED:   "No fingerprint input into fingerprint reader",
    Error.FP_CANCEL:         "Command cancelled",
    Error.RECV_LENGTH:       "Wrong data length",
    Error.RECV_CKS:          "Wrong data check code",
    Error.GATHER_OUT:        "Exceed upper limit of acquisition times",
    Error.RECV_TIMEOUT:      "Data reading timeout",
}


class DFRobot_ID809:
    """
    Python driver cho module vân tay SEN0348 (DFRobot ID809).
    Giao tiếp qua UART serial.
    """

    def __init__(self, port="/dev/ttyTHS1", baudrate=115200, timeout=2.0, debug=False):
        """
        Args:
            port: Cổng serial (Jetson Nano UART2 = /dev/ttyTHS1)
            baudrate: Tốc độ baud mặc định 115200
            timeout: Thời gian chờ đọc (giây)
            debug: Bật chế độ in debug
        """
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._debug = debug
        self._ser = None
        self._error = Error.SUCCESS
        self._number = 0       # Fingerprint acquisition times
        self._state = 0        # Collect fingerprint state
        self.FINGERPRINT_CAPACITY = 80

    def _log(self, *args):
        if self._debug:
            print("[DEBUG]", *args)

    # ============== Serial Connection ==============
    def begin(self):
        """
        Mở cổng serial và kiểm tra kết nối với module.
        Returns:
            True nếu kết nối thành công, False nếu thất bại
        """
        try:
            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self._timeout
            )
            time.sleep(0.1)  # Đợi serial ổn định
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        except serial.SerialException as e:
            print(f"[ERROR] Cannot open serial port {self._port}: {e}")
            return False

        if not self.is_connected():
            print("[ERROR] Module not responding (test connection failed)")
            return False

        # Đọc device info để xác định capacity
        info = self.get_device_info()
        if info:
            if info[-1] == '4':
                self.FINGERPRINT_CAPACITY = 80
            elif info[-1] == '3':
                self.FINGERPRINT_CAPACITY = 200
            self._log(f"Device info: {info}, capacity: {self.FINGERPRINT_CAPACITY}")

        return True

    def close(self):
        """Đóng cổng serial."""
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __del__(self):
        self.close()

    # ============== Protocol Layer ==============
    def _calc_cmd_cks(self, sid, did, cmd, length, payload_bytes):
        """Tính checksum cho command packet."""
        cks = 0xFF
        cks += sid
        cks += did
        cks += (cmd & 0xFF)
        cks += (cmd >> 8)
        cks += (length & 0xFF)
        cks += (length >> 8)
        for b in payload_bytes:
            cks += b
        return cks & 0xFFFF

    def _calc_rcm_cks(self, sid, did, rcm, length, ret, payload_bytes):
        """Tính checksum cho response packet."""
        cks = 0xFF
        cks += sid
        cks += did
        cks += (rcm & 0xFF)
        cks += (rcm >> 8)
        cks += (length & 0xFF)
        cks += (length >> 8)
        cks += (ret & 0xFF)
        cks += (ret >> 8)
        # payload_bytes here is data WITHOUT the 2-byte CKS at end
        for b in payload_bytes:
            cks += b
        return cks & 0xFFFF

    def _pack(self, pkt_type, cmd, payload=None):
        """
        Đóng gói command packet theo giao thức ID809.
        Returns: bytes để gửi qua serial
        """
        if payload is None:
            payload = b''
        if isinstance(payload, (list, tuple)):
            payload = bytes(payload)

        sid = 0
        did = 0
        length = len(payload)

        if pkt_type == CMD_TYPE:
            # Command packet: prefix(2) + SID(1) + DID(1) + CMD(2) + LEN(2) + payload_padded(16) + CKS(2)
            prefix = CMD_PREFIX_CODE
            padded_payload = bytearray(16)
            for i in range(min(len(payload), 16)):
                padded_payload[i] = payload[i]
            cks = self._calc_cmd_cks(sid, did, cmd, length, padded_payload)
            packet = struct.pack('<H', prefix)           # 2 bytes prefix
            packet += struct.pack('<B', sid)             # 1 byte SID
            packet += struct.pack('<B', did)             # 1 byte DID
            packet += struct.pack('<H', cmd)             # 2 bytes CMD
            packet += struct.pack('<H', length)          # 2 bytes LEN
            packet += bytes(padded_payload)              # 16 bytes data
            packet += struct.pack('<H', cks)             # 2 bytes CKS
        else:
            # Data packet: prefix(2) + SID(1) + DID(1) + CMD(2) + LEN(2) + payload(LEN) + CKS(2)
            prefix = CMD_DATA_PREFIX_CODE
            cks = self._calc_cmd_cks(sid, did, cmd, length, payload)
            packet = struct.pack('<H', prefix)
            packet += struct.pack('<B', sid)
            packet += struct.pack('<B', did)
            packet += struct.pack('<H', cmd)
            packet += struct.pack('<H', length)
            packet += payload
            packet += struct.pack('<H', cks)

        return packet

    def _send_packet(self, packet):
        """Gửi packet qua serial."""
        time.sleep(0.01)  # Delay 10ms như Arduino
        self._ser.write(packet)
        self._ser.flush()
        if self._debug:
            self._log("TX:", packet.hex(' '))

    def _read_n(self, n, timeout_ms=2000):
        """Đọc chính xác n bytes từ serial với timeout."""
        data = bytearray()
        start = time.time()
        while len(data) < n:
            remaining = n - len(data)
            chunk = self._ser.read(remaining)
            if chunk:
                data.extend(chunk)
            if (time.time() - start) * 1000 > timeout_ms:
                self._log(f"Read timeout! Got {len(data)}/{n} bytes")
                break
        return bytes(data)

    def _read_prefix(self):
        """
        Đọc prefix và header của response packet.
        Returns: (packet_type, header_dict) hoặc (None, None) nếu timeout
        """
        state = 'INIT'
        pkt_type = None
        start = time.time()

        while True:
            ch_bytes = self._ser.read(1)
            if not ch_bytes:
                if (time.time() - start) > self._timeout:
                    self._log("Read prefix timeout")
                    return None, None
                continue

            ch = ch_bytes[0]

            if state == 'INIT':
                if ch == 0xAA:
                    state = 'AA'
                elif ch == 0xA5:
                    state = 'A5'
            elif state == 'AA':
                if ch == 0x55:
                    pkt_type = RCM_TYPE   # Response packet 0x55AA
                    break
                elif ch == 0xAA:
                    state = 'AA'
                elif ch == 0xA5:
                    state = 'A5'
                else:
                    state = 'INIT'
            elif state == 'A5':
                if ch == 0x5A:
                    pkt_type = DATA_TYPE  # Data packet 0x5AA5
                    break
                elif ch == 0xAA:
                    state = 'AA'
                elif ch == 0xA5:
                    state = 'A5'
                else:
                    state = 'INIT'

            if (time.time() - start) > self._timeout:
                self._log("Read prefix timeout (loop)")
                return None, None

        # Đọc phần còn lại của header: SID(1)+DID(1)+RCM(2)+LEN(2)+RET(2) = 8 bytes
        header_data = self._read_n(8)
        if len(header_data) < 8:
            return None, None

        sid = header_data[0]
        did = header_data[1]
        rcm = struct.unpack('<H', header_data[2:4])[0]
        length = struct.unpack('<H', header_data[4:6])[0]
        ret = struct.unpack('<H', header_data[6:8])[0]

        if pkt_type == RCM_TYPE:
            prefix = RCM_PREFIX_CODE
        else:
            prefix = RCM_DATA_PREFIX_CODE

        header = {
            'prefix': prefix,
            'sid': sid,
            'did': did,
            'rcm': rcm,
            'len': length,
            'ret': ret,
        }
        return pkt_type, header

    def _response_payload(self):
        """
        Đọc response packet hoàn chỉnh.
        Returns: (error_code, payload_data)
            error_code: ERR_SUCCESS hoặc ERR_ID809
            payload_data: bytes dữ liệu payload (không bao gồm CKS)
        """
        pkt_type, header = self._read_prefix()
        if pkt_type is None:
            self._error = Error.RECV_TIMEOUT
            return ERR_ID809, b''

        if pkt_type == RCM_TYPE:
            # Response packet: 14 bytes data + 2 bytes CKS
            data_len = 14 + 2
        else:
            # Data packet: (LEN) bytes bao gồm CKS
            data_len = header['len']

        payload_raw = self._read_n(data_len)
        if len(payload_raw) < data_len:
            self._error = Error.RECV_LENGTH
            return ERR_ID809, b''

        # CKS là 2 byte cuối
        cks_received = struct.unpack('<H', payload_raw[-2:])[0]
        payload_data = payload_raw[:-2]  # Dữ liệu không bao gồm CKS

        ret_val = header['ret'] & 0xFF
        self._error = Error(ret_val) if ret_val in Error._value2member_map_ else Error.FAIL

        if ret_val != ERR_SUCCESS:
            self._log(f"Response error: 0x{ret_val:02X} - {self.get_error_description()}")
            return ERR_ID809, payload_data

        # Verify checksum
        expected_cks = self._calc_rcm_cks(
            header['sid'], header['did'], header['rcm'],
            header['len'], header['ret'], payload_data
        )
        if expected_cks != cks_received:
            self._log(f"CKS mismatch: expected=0x{expected_cks:04X}, got=0x{cks_received:04X}")
            self._error = Error.RECV_CKS
            return ERR_ID809, payload_data

        if self._debug:
            self._log(f"RX OK: type=0x{pkt_type:02X}, ret=0x{ret_val:02X}, data={payload_data.hex(' ')}")

        return ERR_SUCCESS, payload_data

    def _send_and_receive(self, pkt_type, cmd, payload=None):
        """
        Gửi command và nhận response.
        Returns: (error_code, payload_data)
        """
        packet = self._pack(pkt_type, cmd, payload)
        self._send_packet(packet)
        return self._response_payload()

    # ============== Public API ==============

    def is_connected(self):
        """
        Kiểm tra kết nối với module.
        Returns: True nếu module phản hồi OK
        """
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_TEST_CONNECTION)
        return ret == ERR_SUCCESS

    def get_device_info(self):
        """
        Đọc thông tin thiết bị.
        Returns: Chuỗi thông tin hoặc "" nếu lỗi
        """
        ret, data = self._send_and_receive(CMD_TYPE, CMD_DEVICE_INFO)
        if ret != ERR_SUCCESS:
            return ""
        # Response đầu tiên chứa độ dài data
        data_len = data[0] + (data[1] << 8) if len(data) >= 2 else 0
        if data_len == 0:
            return ""
        # Đọc data packet tiếp theo
        ret2, info_data = self._response_payload()
        if ret2 != ERR_SUCCESS:
            return ""
        try:
            return info_data[:data_len].decode('ascii', errors='ignore').rstrip('\x00')
        except:
            return ""

    def get_device_id(self):
        """Đọc device ID (1-255)."""
        return self._get_param(0)

    def get_security_level(self):
        """Đọc security level (1-5)."""
        return self._get_param(1)

    def get_duplication_check(self):
        """Đọc trạng thái kiểm tra trùng lặp (0/1)."""
        return self._get_param(2)

    def get_baudrate(self):
        """Đọc baudrate hiện tại."""
        return self._get_param(3)

    def get_self_learn(self):
        """Đọc trạng thái self-learn (0/1)."""
        return self._get_param(4)

    def set_device_id(self, device_id):
        """Đặt device ID (1-255)."""
        return self._set_param(0, device_id)

    def set_security_level(self, level):
        """Đặt security level (1-5)."""
        return self._set_param(1, level)

    def set_duplication_check(self, enable):
        """Bật/tắt kiểm tra trùng lặp (0/1)."""
        return self._set_param(2, enable)

    def set_baudrate(self, baudrate):
        """Đặt baudrate (DeviceBaudrate enum)."""
        return self._set_param(3, baudrate)

    def set_self_learn(self, enable):
        """Bật/tắt self-learn (0/1)."""
        return self._set_param(4, enable)

    def _set_param(self, param_type, value):
        data = bytes([param_type, value, 0, 0, 0])
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_SET_PARAM, data)
        return ret

    def _get_param(self, param_type):
        data = bytes([param_type])
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_GET_PARAM, data)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return ERR_ID809

    def get_module_sn(self):
        """Đọc số serial module."""
        ret, data = self._send_and_receive(CMD_TYPE, CMD_GET_MODULE_SN)
        if ret != ERR_SUCCESS:
            return ""
        data_len = data[0] + (data[1] << 8) if len(data) >= 2 else 0
        if data_len == 0:
            return ""
        ret2, sn_data = self._response_payload()
        if ret2 != ERR_SUCCESS:
            return ""
        try:
            return sn_data[:data_len].decode('ascii', errors='ignore').rstrip('\x00')
        except:
            return ""

    def ctrl_led(self, mode, color, blink_count=0):
        """
        Điều khiển LED.
        Args:
            mode: LEDMode enum
            color: LEDColor enum
            blink_count: Số lần nhấp nháy (0 = liên tục)
        """
        data = bytearray(4)
        if self.FINGERPRINT_CAPACITY == 80:
            data[0] = mode
            data[1] = color
            data[2] = color
            data[3] = blink_count
        else:
            mode_map = {1: 2, 2: 4, 3: 1, 4: 0, 5: 3}
            data[0] = mode_map.get(mode, 0)
            color_map = {
                LEDColor.GREEN:   0x84,
                LEDColor.RED:     0x82,
                LEDColor.YELLOW:  0x86,
                LEDColor.BLUE:    0x81,
                LEDColor.CYAN:    0x85,
                LEDColor.MAGENTA: 0x83,
                LEDColor.WHITE:   0x87,
            }
            c = color_map.get(color, 0x87)
            data[1] = c
            data[2] = c
            data[3] = blink_count

        ret, _ = self._send_and_receive(CMD_TYPE, CMD_SLED_CTRL, bytes(data))
        return ret

    def detect_finger(self):
        """
        Kiểm tra có ngón tay đặt trên cảm biến không.
        Returns: 1 (có) hoặc 0 (không)
        """
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_FINGER_DETECT)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return 0

    def get_empty_id(self):
        """Lấy ID đầu tiên còn trống để đăng ký."""
        data = bytes([1, 0, self.FINGERPRINT_CAPACITY, 0])
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_GET_EMPTY_ID, data)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return ERR_ID809

    def get_status_id(self, fp_id):
        """Kiểm tra ID đã đăng ký chưa (0=đã đăng ký, 1=chưa)."""
        data = bytes([fp_id, 0])
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_GET_STATUS, data)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return ERR_ID809

    def get_enroll_count(self):
        """Lấy số lượng vân tay đã đăng ký."""
        data = bytes([1, 0, self.FINGERPRINT_CAPACITY, 0])
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_GET_ENROLL_COUNT, data)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return ERR_ID809

    def collection_fingerprint(self, timeout_s=10, ram_number=-1):
        """
        Thu thập vân tay: chờ ngón tay -> chụp ảnh -> tạo template.
        Args:
            timeout_s: Thời gian chờ (giây), 0 = chờ vô hạn
            ram_number: Chỉ định RAM buffer (-1 = tự động)
        Returns: ERR_SUCCESS hoặc ERR_ID809
        """
        if ram_number == -1:
            if self._number > 2:
                self._error = Error.GATHER_OUT
                return ERR_ID809

        # Chờ ngón tay
        start = time.time()
        while not self.detect_finger():
            time.sleep(0.01)
            if timeout_s > 0 and (time.time() - start) > timeout_s:
                self._error = Error.TIMEOUT
                self._state = 0
                return ERR_ID809

        # Chụp ảnh
        ret = self._get_image()
        if ret != ERR_SUCCESS:
            self._state = 0
            return ERR_ID809

        # Tạo template
        buf_id = ram_number if ram_number != -1 else self._number
        ret = self._generate(buf_id)
        if ret != ERR_SUCCESS:
            self._state = 0
            return ERR_ID809

        self._number += 1
        self._state = 1
        return ret

    def store_fingerprint(self, fp_id):
        """
        Lưu vân tay đã thu thập vào ID.
        Args:
            fp_id: ID để lưu (1-80 hoặc 1-200)
        """
        ret = self._merge()
        if ret != ERR_SUCCESS:
            return ERR_ID809
        self._number = 0
        data = bytes([fp_id, 0, 0, 0])
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_STORE_CHAR, data)
        return ret

    def del_fingerprint(self, fp_id):
        """
        Xóa vân tay. fp_id=DELALL (0xFF) để xóa tất cả.
        """
        data = bytearray(4)
        if fp_id == DELALL:
            data[0] = 1
            data[2] = self.FINGERPRINT_CAPACITY
        else:
            data[0] = fp_id
            data[2] = fp_id
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_DEL_CHAR, bytes(data))
        return ret

    def search(self):
        """
        So sánh vân tay vừa thu thập với toàn bộ thư viện (1:N).
        Returns: ID khớp, 0 nếu không khớp, ERR_ID809 nếu lỗi
        """
        if self._state != 1:
            return 0
        data = bytes([0, 0, 1, 0, self.FINGERPRINT_CAPACITY, 0])
        self._number = 0
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_SEARCH, data)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return 0

    def verify(self, fp_id):
        """
        So sánh vân tay vừa thu thập với ID cụ thể (1:1).
        Returns: ID khớp, 0 nếu không khớp
        """
        if self._state != 1:
            return 0
        data = bytes([fp_id, 0, 0, 0])
        self._number = 0
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_VERIFY, data)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return 0

    def match(self, ram_buffer_id0, ram_buffer_id1):
        """So sánh 2 template trong RAM buffer."""
        data = bytes([ram_buffer_id0, 0, ram_buffer_id1, 0])
        ret, payload = self._send_and_receive(CMD_TYPE, CMD_MATCH, data)
        if ret == ERR_SUCCESS and len(payload) > 0:
            return payload[0]
        return 0

    def load_fingerprint(self, fp_id, ram_buffer_id):
        """Load vân tay từ thư viện vào RAM buffer."""
        data = bytes([fp_id, 0, ram_buffer_id, 0])
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_LOAD_CHAR, data)
        return ret

    def get_quarter_finger_image(self):
        """
        Chụp và upload ảnh vân tay 1/4 (80x80 pixels, grayscale 8-bit).
        Returns: bytearray 6400 bytes ảnh, hoặc None nếu lỗi
        """
        return self._get_finger_image_internal(quarter=True)

    def get_finger_image(self):
        """
        Chụp và upload ảnh vân tay đầy đủ (160x160 pixels, grayscale 8-bit).
        Returns: bytearray 25600 bytes ảnh, hoặc None nếu lỗi
        """
        return self._get_finger_image_internal(quarter=False)

    def _get_finger_image_internal(self, quarter=True):
        """Nội bộ: chụp ảnh và nhận dữ liệu pixel từ module."""
        # Bước 1: Chụp ảnh
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_GET_IMAGE)
        if ret != ERR_SUCCESS:
            self._log("GET_IMAGE failed")
            return None

        # Bước 2: Yêu cầu upload ảnh (0=full, 1=quarter)
        img_type = bytes([1]) if quarter else bytes([0])
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_UP_IMAGE_CODE, img_type)
        if ret != ERR_SUCCESS:
            self._log("UP_IMAGE_CODE failed")
            return None

        # Bước 3: Nhận các data packet chứa pixel
        if quarter:
            num_packets = 13
            total_size = 6400    # 80x80
            last_packet_size = 448
        else:
            num_packets = 52
            total_size = 25600   # 160x160
            last_packet_size = 304

        image = bytearray(total_size)
        offset = 0
        for i in range(num_packets):
            ret_code, pkt_data = self._response_payload()
            if ret_code != ERR_SUCCESS:
                self._log(f"Image packet {i} failed")
                return None
            # Bỏ 2 byte đầu (index), lấy phần pixel data
            pixel_data = pkt_data[2:] if len(pkt_data) > 2 else pkt_data
            if i == num_packets - 1:
                copy_len = last_packet_size
            else:
                copy_len = 496
            actual_len = min(copy_len, len(pixel_data))
            image[offset:offset + actual_len] = pixel_data[:actual_len]
            offset += 496  # Stride cố định 496 như code C++

        return image

    def enter_standby_state(self):
        """Đưa module vào chế độ ngủ."""
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_ENTER_STANDBY_STATE)
        return ret

    def get_error_description(self):
        """Lấy mô tả lỗi cuối cùng."""
        return ERROR_DESCRIPTIONS.get(self._error, f"Unknown error: 0x{self._error:02X}")

    @property
    def error_code(self):
        return self._error

    # ============== Internal Commands ==============
    def _get_image(self):
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_GET_IMAGE)
        return ret

    def _generate(self, ram_buffer_id):
        data = bytes([ram_buffer_id, 0])
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_GENERATE, data)
        return ret

    def _merge(self):
        data = bytes([0, 0, self._number])
        ret, _ = self._send_and_receive(CMD_TYPE, CMD_MERGE, data)
        return ret
