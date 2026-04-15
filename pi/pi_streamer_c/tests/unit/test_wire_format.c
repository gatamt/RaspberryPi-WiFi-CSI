#include "pi_streamer/wire_format.h"
#include "unity.h"

#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static void test_constants_are_correct(void) {
    TEST_ASSERT_EQUAL_UINT(28u,   PI_CHUNK_HEADER_SIZE);
    TEST_ASSERT_EQUAL_UINT(1400u, PI_MAX_PAYLOAD_SIZE);
    TEST_ASSERT_EQUAL_UINT(1428u, PI_MAX_DATAGRAM_SIZE);
}

// Exact byte-for-byte fixture lifted from the Python reference's
// wire_format._self_test. This is the primary contract: Python and C MUST
// serialize identically or the existing iOS client breaks.
static void test_pack_matches_python_reference(void) {
    const pi_chunk_header_t hdr = {
        .frame_id     = 0x11223344u,
        .width        = 1280,
        .height       = 720,
        .timestamp_ms = 0xAABBCCDDu,
        .total_len    = 0x00010000u,
        .chunk_idx    = 2,
        .chunk_count  = 73,
        .frame_type   = 1,
        .reserved     = 0,
    };
    uint8_t buf[PI_CHUNK_HEADER_SIZE] = {0};
    pi_chunk_header_pack(&hdr, buf);

    static const uint8_t expected[PI_CHUNK_HEADER_SIZE] = {
        0x44, 0x33, 0x22, 0x11,             // frame_id
        0x00, 0x05,                         // width  (1280 = 0x0500)
        0xD0, 0x02,                         // height (720  = 0x02D0)
        0xDD, 0xCC, 0xBB, 0xAA,             // timestamp_ms
        0x00, 0x00, 0x01, 0x00,             // total_len
        0x02, 0x00,                         // chunk_idx
        0x49, 0x00,                         // chunk_count (73 = 0x49)
        0x01, 0x00, 0x00, 0x00,             // frame_type
        0x00, 0x00, 0x00, 0x00,             // reserved
    };
    TEST_ASSERT_EQUAL_MEMORY(expected, buf, PI_CHUNK_HEADER_SIZE);
}

static void test_pack_unpack_round_trip(void) {
    const pi_chunk_header_t hdr_in = {
        .frame_id     = 42u,
        .width        = 1280,
        .height       = 720,
        .timestamp_ms = 1234567u,
        .total_len    = 8000u,
        .chunk_idx    = 5,
        .chunk_count  = 6,
        .frame_type   = 0,
        .reserved     = 0xDEADBEEFu,
    };
    uint8_t buf[PI_CHUNK_HEADER_SIZE];
    pi_chunk_header_pack(&hdr_in, buf);

    pi_chunk_header_t hdr_out;
    TEST_ASSERT_EQUAL_INT(0, pi_chunk_header_unpack(buf, &hdr_out));
    TEST_ASSERT_EQUAL_UINT32(hdr_in.frame_id,     hdr_out.frame_id);
    TEST_ASSERT_EQUAL_UINT16(hdr_in.width,        hdr_out.width);
    TEST_ASSERT_EQUAL_UINT16(hdr_in.height,       hdr_out.height);
    TEST_ASSERT_EQUAL_UINT32(hdr_in.timestamp_ms, hdr_out.timestamp_ms);
    TEST_ASSERT_EQUAL_UINT32(hdr_in.total_len,    hdr_out.total_len);
    TEST_ASSERT_EQUAL_UINT16(hdr_in.chunk_idx,    hdr_out.chunk_idx);
    TEST_ASSERT_EQUAL_UINT16(hdr_in.chunk_count,  hdr_out.chunk_count);
    TEST_ASSERT_EQUAL_UINT32(hdr_in.frame_type,   hdr_out.frame_type);
    TEST_ASSERT_EQUAL_UINT32(hdr_in.reserved,     hdr_out.reserved);
}

static void test_unpack_null_returns_error(void) {
    pi_chunk_header_t hdr;
    uint8_t buf[PI_CHUNK_HEADER_SIZE] = {0};
    TEST_ASSERT_EQUAL_INT(-1, pi_chunk_header_unpack(NULL, &hdr));
    TEST_ASSERT_EQUAL_INT(-1, pi_chunk_header_unpack(buf, NULL));
}

static void test_chunk_count_edges(void) {
    TEST_ASSERT_EQUAL_UINT(0u, pi_chunk_count_for(0));
    TEST_ASSERT_EQUAL_UINT(1u, pi_chunk_count_for(1));
    TEST_ASSERT_EQUAL_UINT(1u, pi_chunk_count_for(1400));
    TEST_ASSERT_EQUAL_UINT(2u, pi_chunk_count_for(1401));
    TEST_ASSERT_EQUAL_UINT(2u, pi_chunk_count_for(2800));
    TEST_ASSERT_EQUAL_UINT(3u, pi_chunk_count_for(2801));
    TEST_ASSERT_EQUAL_UINT(3u, pi_chunk_count_for(3000));  // matches Python
}

static void test_build_one_chunk_three_way_split(void) {
    // Python reference: 3000 bytes -> 3 chunks of 1400 / 1400 / 200.
    uint8_t frame[3000];
    for (size_t i = 0; i < sizeof frame; ++i) {
        frame[i] = (uint8_t)(i & 0xFFu);
    }

    uint8_t buf[PI_MAX_DATAGRAM_SIZE];
    pi_chunk_header_t hdr;

    // --- chunk 0 ---
    size_t n = pi_build_one_chunk(1u, 1280, 720, 42u,
                                  frame, sizeof frame, 0,
                                  /*is_keyframe=*/true,
                                  buf, sizeof buf);
    TEST_ASSERT_EQUAL_UINT(PI_CHUNK_HEADER_SIZE + 1400u, n);
    TEST_ASSERT_EQUAL_INT(0, pi_chunk_header_unpack(buf, &hdr));
    TEST_ASSERT_EQUAL_UINT32(1u,    hdr.frame_id);
    TEST_ASSERT_EQUAL_UINT32(3000u, hdr.total_len);
    TEST_ASSERT_EQUAL_UINT16(0,     hdr.chunk_idx);
    TEST_ASSERT_EQUAL_UINT16(3,     hdr.chunk_count);
    TEST_ASSERT_EQUAL_UINT32(1u,    hdr.frame_type);
    TEST_ASSERT_EQUAL_UINT32(42u,   hdr.timestamp_ms);
    TEST_ASSERT_EQUAL_MEMORY(frame, buf + PI_CHUNK_HEADER_SIZE, 1400);

    // --- chunk 1 ---
    n = pi_build_one_chunk(1u, 1280, 720, 42u,
                           frame, sizeof frame, 1,
                           true, buf, sizeof buf);
    TEST_ASSERT_EQUAL_UINT(PI_CHUNK_HEADER_SIZE + 1400u, n);
    TEST_ASSERT_EQUAL_INT(0, pi_chunk_header_unpack(buf, &hdr));
    TEST_ASSERT_EQUAL_UINT16(1, hdr.chunk_idx);
    TEST_ASSERT_EQUAL_MEMORY(frame + 1400,
                             buf + PI_CHUNK_HEADER_SIZE,
                             1400);

    // --- chunk 2 (short) ---
    n = pi_build_one_chunk(1u, 1280, 720, 42u,
                           frame, sizeof frame, 2,
                           true, buf, sizeof buf);
    TEST_ASSERT_EQUAL_UINT(PI_CHUNK_HEADER_SIZE + 200u, n);
    TEST_ASSERT_EQUAL_INT(0, pi_chunk_header_unpack(buf, &hdr));
    TEST_ASSERT_EQUAL_UINT16(2, hdr.chunk_idx);
    TEST_ASSERT_EQUAL_MEMORY(frame + 2800,
                             buf + PI_CHUNK_HEADER_SIZE,
                             200);
}

static void test_build_one_chunk_exact_1400_is_one_chunk(void) {
    uint8_t frame[1400];
    memset(frame, 0xAB, sizeof frame);

    uint8_t buf[PI_MAX_DATAGRAM_SIZE];
    size_t n = pi_build_one_chunk(7u, 1, 1, 0,
                                  frame, sizeof frame, 0,
                                  false, buf, sizeof buf);
    TEST_ASSERT_EQUAL_UINT(PI_CHUNK_HEADER_SIZE + 1400u, n);

    pi_chunk_header_t hdr;
    pi_chunk_header_unpack(buf, &hdr);
    TEST_ASSERT_EQUAL_UINT16(1, hdr.chunk_count);
    TEST_ASSERT_EQUAL_UINT32(0u, hdr.frame_type);  // P-frame
}

static void test_build_one_chunk_rejects_bad_inputs(void) {
    uint8_t frame[10] = {0};
    uint8_t buf[PI_MAX_DATAGRAM_SIZE];

    // NULL frame
    TEST_ASSERT_EQUAL_UINT(0, pi_build_one_chunk(
        1, 0, 0, 0, NULL, 10, 0, false, buf, sizeof buf));
    // NULL out_buf
    TEST_ASSERT_EQUAL_UINT(0, pi_build_one_chunk(
        1, 0, 0, 0, frame, 10, 0, false, NULL, 1428));
    // zero-size frame
    TEST_ASSERT_EQUAL_UINT(0, pi_build_one_chunk(
        1, 0, 0, 0, frame, 0, 0, false, buf, sizeof buf));
    // chunk_idx out of range (10 bytes -> 1 chunk)
    TEST_ASSERT_EQUAL_UINT(0, pi_build_one_chunk(
        1, 0, 0, 0, frame, 10, 1, false, buf, sizeof buf));
    // out_cap too small for even the header
    TEST_ASSERT_EQUAL_UINT(0, pi_build_one_chunk(
        1, 0, 0, 0, frame, 10, 0, false, buf, 10));
    // out_cap fits header but not payload
    TEST_ASSERT_EQUAL_UINT(0, pi_build_one_chunk(
        1, 0, 0, 0, frame, 10, 0, false, buf, PI_CHUNK_HEADER_SIZE + 5));
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_constants_are_correct);
    RUN_TEST(test_pack_matches_python_reference);
    RUN_TEST(test_pack_unpack_round_trip);
    RUN_TEST(test_unpack_null_returns_error);
    RUN_TEST(test_chunk_count_edges);
    RUN_TEST(test_build_one_chunk_three_way_split);
    RUN_TEST(test_build_one_chunk_exact_1400_is_one_chunk);
    RUN_TEST(test_build_one_chunk_rejects_bad_inputs);
    return UNITY_END();
}
