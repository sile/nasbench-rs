use crate::Result;
use byteorder::{ByteOrder as _, LittleEndian, ReadBytesExt as _};
use crc;
use std::io::Read;
use trackable::error::Failed;

/// https://www.tensorflow.org/tutorials/load_data/tf_records
#[derive(Debug)]
pub struct TfRecord {
    pub len: u64,
    pub data: Vec<u8>,
}
impl TfRecord {
    pub fn read_from<R: Read>(mut reader: R) -> Result<Self> {
        let mut len_buf = [0; 8];
        track_any_err!(reader.read_exact(&mut len_buf))?;
        let len = LittleEndian::read_u64(&len_buf);
        let len_crc = track_any_err!(reader.read_u32::<LittleEndian>())?;
        track!(check_crc(&len_buf, len_crc))?;

        let mut data = vec![0; len as usize];
        track_any_err!(reader.read_exact(&mut data))?;
        let data_crc = track_any_err!(reader.read_u32::<LittleEndian>())?;
        track!(check_crc(&data, data_crc))?;

        Ok(Self { len, data })
    }
}

#[derive(Debug)]
pub struct TfRecordStream<R> {
    reader: R,
}
impl<R: Read> TfRecordStream<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}
impl<R: Read> Iterator for TfRecordStream<R> {
    type Item = Result<TfRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut peek = [0; 1];
        match track_any_err!(self.reader.read(&mut peek)) {
            Err(e) => Some(Err(e)),
            Ok(0) => None,
            Ok(_) => match track!(TfRecord::read_from(peek.chain(&mut self.reader))) {
                Err(e) => Some(Err(e)),
                Ok(r) => Some(Ok(r)),
            },
        }
    }
}

fn check_crc(bytes: &[u8], actual_crc: u32) -> Result<()> {
    let expected_crc = crc::crc32::checksum_castagnoli(&bytes);
    let expected_crc = (expected_crc.overflowing_shr(15).0 | expected_crc.overflowing_shl(17).0)
        .overflowing_add(0xa282ead8)
        .0;
    track_assert_eq!(actual_crc, expected_crc, Failed);
    Ok(())
}
