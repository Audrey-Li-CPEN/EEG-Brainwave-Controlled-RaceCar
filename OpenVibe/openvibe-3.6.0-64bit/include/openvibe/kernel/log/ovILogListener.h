#pragma once

#include "../ovIKernelObject.h"

namespace OpenViBE {
namespace Kernel {

/// <summary> Log Levels. </summary>
enum ELogLevel
{
	LogLevel_First,
	LogLevel_None,
	LogLevel_Debug,
	LogLevel_Benchmark,
	LogLevel_Trace,
	LogLevel_Info,
	LogLevel_Warning,
	LogLevel_ImportantWarning,
	LogLevel_Error,
	LogLevel_Fatal,
	LogLevel_Last,
};

inline std::string toString(const ELogLevel level)
{
	switch (level)
	{
		case LogLevel_First: return "[ FIRST ] ";
		case LogLevel_None: return "[ NONE  ] ";
		case LogLevel_Debug: return "[ DEBUG ] ";
		case LogLevel_Benchmark: return "[ BENCH ] ";
		case LogLevel_Trace: return "[ TRACE ] ";
		case LogLevel_Info: return "[  INF  ] ";
		case LogLevel_Warning: return "[WARNING] ";
		case LogLevel_ImportantWarning: return "[WARNING] ";
		case LogLevel_Error: return "[ ERROR ] ";
		case LogLevel_Fatal: return "[ FATAL ] ";
		case LogLevel_Last: return "[ LAST  ] ";
		default: return "[UNKNOWN] ";
	}
}

/// <summary> Log colors. </summary>
enum ELogColor
{
	LogColor_ForegroundColorRedBit = 0x00000001,
	LogColor_ForegroundColorGreenBit = 0x00000002,
	LogColor_ForegroundColorBlueBit = 0x00000004,
	LogColor_ForegroundColorBit = 0x00000008,
	LogColor_ForegroundLightStateBit = 0x00000010,
	LogColor_ForegroundLightBit = 0x00000020,
	LogColor_ForegroundBlinkStateBit = 0x00000040,
	LogColor_ForegroundBlinkBit = 0x00000080,
	LogColor_ForegroundBoldStateBit = 0x00000100,
	LogColor_ForegroundBoldBit = 0x00000200,
	LogColor_ForegroundUnderlineStateBit = 0x00000400,
	LogColor_ForegroundUnderlineBit = 0x00000800,

	LogColor_BackgroundColorRedBit = 0x00001000,
	LogColor_BackgroundColorGreenBit = 0x00002000,
	LogColor_BackgroundColorBlueBit = 0x00004000,
	LogColor_BackgroundColorBit = 0x00008000,
	LogColor_BackgroundLightStateBit = 0x00010000,
	LogColor_BackgroundLightBit = 0x00020000,
	LogColor_BackgroundBlinkStateBit = 0x00040000,
	LogColor_BackgroundBlinkBit = 0x00080000,

	LogColor_ForegroundBit = 0x00100000,
	LogColor_BackgroundBit = 0x00200000,
	LogColor_PushStateBit = 0x00400000,
	LogColor_PopStateBit = 0x00800000,
	LogColor_ResetBit = 0x08000000,

	LogColor_Default = 0x00000000,

	LogColor_ForegroundBlack = 0x00100008,
	LogColor_ForegroundRed = 0x00100009,
	LogColor_ForegroundGreen = 0x0010000A,
	LogColor_ForegroundYellow = 0x0010000B,
	LogColor_ForegroundBlue = 0x0010000C,
	LogColor_ForegroundMagenta = 0x0010000D,
	LogColor_ForegroundCyan = 0x0010000E,
	LogColor_ForegroundWhite = 0x0010000F,
	LogColor_ForegroundLightOff = 0x00100020,
	LogColor_ForegroundLightOn = 0x00100030,
	LogColor_ForegroundBlinkOff = 0x00100080,
	LogColor_ForegroundBlinkOn = 0x001000C0,
	LogColor_ForegroundBoldOff = 0x00100200,
	LogColor_ForegroundBoldOn = 0x00100300,
	LogColor_ForegroundUnderlineOff = 0x00100800,
	LogColor_ForegroundUnderlineOn = 0x00100C00,

	LogColor_BackgroundBlack = 0x00208000,
	LogColor_BackgroundRed = 0x00209000,
	LogColor_BackgroundGreen = 0x0020A000,
	LogColor_BackgroundYellow = 0x0020B000,
	LogColor_BackgroundBlue = 0x0020C000,
	LogColor_BackgroundMagenta = 0x0020D000,
	LogColor_BackgroundCyan = 0x0020E000,
	LogColor_BackgroundWhite = 0x0020F000,
	LogColor_BackgroundLightOff = 0x00220000,
	LogColor_BackgroundLightOn = 0x00230000,
	LogColor_BackgroundBlinkOff = 0x00280000,
	LogColor_BackgroundBlinkOn = 0x002C0000,
};

/**
 * \class ILogListener
 * \brief Log manager's listener interface
 * \author Yann Renard (INRIA/IRISA)
 * \date 2006-06-03
 * \ingroup Group_Log
 * \ingroup Group_Kernel
 *
 * The log listener is derived and implemented such as it can
 * effectively process the log action. It could do it in a
 * file, in a notification area, or whatever fits your needs.
 * Objects to log are sent to it thanks to the log manager
 * after a listener has been registered to it.
 */
class OV_API ILogListener : public IKernelObject
{
public:

	/** \name Log level activation */
	//@{

	/**
	 * \brief Tests whether a log level is active or not
	 * \param level [in] : the log level which has to be tested
	 * \return \e true if this log level is active.
	 * \return \e false if this log level is not active.
	 */
	virtual bool isActive(const ELogLevel level) = 0;
	/**
	 * \brief Changes the activation status of a specific log level
	 * \param level [in] : the log level which status has to be changed
	 * \param active [in] : a boolean telling whether this level should be active or not
	 * \return \e true in case of success.
	 * \return \e false in case of error.
	 */
	virtual bool activate(const ELogLevel level, const bool active) = 0;
	/**
	 * \brief Changes the activation status of a specific range of log level
	 * \param startLogLevel [in] : the first log level which status has to be changed
	 * \param endLogLevel [in] : the last log level which status has to be changed
	 * \param active [in] : a boolean telling whether these levels should be active or not
	 * \return \e true in case of success.
	 * \return \e false in case of error.
	 */
	virtual bool activate(const ELogLevel startLogLevel, const ELogLevel endLogLevel, const bool active) = 0;
	/**
	 * \brief Changes the activation status of all log levels at once
	 * \param active [in] : a boolean telling whether the levels should be active or not
	 * \return \e true in case of success.
	 * \return \e false in case of error.
	 */
	virtual bool activate(const bool active) = 0;

	//@}
	/** \name Logging function */
	//@{

	/**
	 * \brief Logs a formatted time value (64 bits unsigned integer)
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const CTime value) = 0;
#if defined __clang__
	/**
	 * \brief Logs a size_t unsigned integer
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const size_t value) = 0;
#endif
	/**
	 * \brief Logs a 64 bits unsigned integer
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const uint64_t value) = 0;
	/**
	 * \brief Logs a 32 bits unsigned integer
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const uint32_t value) = 0;
	/**
	 * \brief Logs a 64 bits signed integer
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const int64_t value) = 0;
	/**
	 * \brief Logs a 32 bits signed integer
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const int value) = 0;
	/**
	 * \brief Logs a 64 bits floating point value
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const double value) = 0;
	/**
	 * \brief Logs a boolean value
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const bool value) = 0;
	/**
	 * \brief Logs an identifier value
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const CIdentifier& value) = 0;
	/**
	 * \brief Logs an OpenViBE string value
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const CString& value) = 0;
	/**
	 * \brief Logs a string
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const std::string& value) = 0;
	/**
	 * \brief Logs an ASCII string value
	 * \param value [in] : the value that should be logged
	 */
	virtual void log(const char* value) = 0;

	//@}
	/** \name Manipulators */
	//@{

	/**
	 * \brief Changes the log level
	 * \param level [in] : the new log level
	 */
	virtual void log(const ELogLevel level) = 0;
	/**
	 * \brief Changes the log color
	 * \param color [in] : the new log color
	 */
	virtual void log(const ELogColor color) = 0;

	//@}

	_IsDerivedFromClass_(IKernelObject, OV_ClassId_Kernel_Log_LogListener)
};

}  // namespace Kernel
}  // namespace OpenViBE
