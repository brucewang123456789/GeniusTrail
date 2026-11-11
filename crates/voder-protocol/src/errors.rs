//! Stable, serializable protocol failures. Never expose internal stack traces or secrets.
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    InvalidJson,
    InvalidFrame,
    InvalidField,
    InvalidPayload,
    UnsupportedVersion,
    UnsupportedMethod,
    InvalidTransition,
    DuplicateOperation,
    FrameTooLarge,
    Unauthorized,
    NotFound,
    Conflict,
    Internal,
}
impl ErrorCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidJson => "INVALID_JSON",
            Self::InvalidFrame => "INVALID_FRAME",
            Self::InvalidField => "INVALID_FIELD",
            Self::InvalidPayload => "INVALID_PAYLOAD",
            Self::UnsupportedVersion => "UNSUPPORTED_VERSION",
            Self::UnsupportedMethod => "UNSUPPORTED_METHOD",
            Self::InvalidTransition => "INVALID_TRANSITION",
            Self::DuplicateOperation => "DUPLICATE_OPERATION",
            Self::FrameTooLarge => "FRAME_TOO_LARGE",
            Self::Unauthorized => "UNAUTHORIZED",
            Self::NotFound => "NOT_FOUND",
            Self::Conflict => "CONFLICT",
            Self::Internal => "INTERNAL",
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtocolError {
    pub code: ErrorCode,
    pub message: String,
}
impl ProtocolError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}
impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}
impl std::error::Error for ProtocolError {}
pub type Result<T> = std::result::Result<T, ProtocolError>;

pub(crate) fn err(code: ErrorCode, message: &'static str) -> ProtocolError {
    ProtocolError::new(code, message)
}
