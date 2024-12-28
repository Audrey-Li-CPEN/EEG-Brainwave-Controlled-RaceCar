#pragma once

#include "ovILogListener.h"

namespace OpenViBE {
namespace Kernel {
/**
 * \class ILogManager
 * \brief Log manager
 * \author Yann Renard (INRIA/IRISA)
 * \date 2006-06-03
 * \ingroup Group_Log
 * \ingroup Group_Kernel
 *
 * The log manager is responsible for keeping a trace of all the messages the application could send as debug output. Such information is not
 * useful most of the cases but could become crucial in some cases. Thus there are different levels of activation for the log manager
 * to work. The log manager forwards each log request to its registered log listeners that effectively do the log the way they want
 * (be it a status window, a console, a file, whatever). See ILogListener for more details.
 */
class OV_API ILogManager : public ILogListener
{
public:

	/**
	 * \brief Registers a new log listener
	 * \param listener [in] : the new listener to register
	 * \return \e true in case of success. \e false in case of error.
	 */
	virtual bool addListener(ILogListener* listener) = 0;
	/**
	 * \brief Removes a registered listener
	 * \param listener [in] : the listener to unregister
	 * \return \e true in case of success. \e false in case of error.
	 */
	virtual bool removeListener(ILogListener* listener) = 0;

	_IsDerivedFromClass_(ILogListener, OV_ClassId_Kernel_Log_LogManager)
};

/**
 * \brief Stream output operator
 * \param logManager [in] : the log manager that takes the object
 * \param object [in] : the object to log
 * \return The log manager itself
 * \sa ILogManager
 *
 * This function helps in logging different objects thanks to the
 * stream operator. The log manager can almost be used as any std
 * ostream object.
 */

#if defined __clang__
inline ILogManager& operator <<(ILogManager& logManager, const size_t object)
{
  logManager.log(object);
  return logManager;
}
#endif

inline ILogManager& operator <<(ILogManager& logManager, const uint64_t object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const uint32_t object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const int64_t object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const int object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const double object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const bool object)
{
  logManager.log(object);
  return logManager;
}
 
inline ILogManager& operator <<(ILogManager& logManager, const CTime object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const CIdentifier& object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const CString& object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const std::string& object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const char* object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const ELogColor object)
{
  logManager.log(object);
  return logManager;
}
inline ILogManager& operator <<(ILogManager& logManager, const ELogLevel object)
{
  logManager.log(object);
  return logManager;
}

}  // namespace Kernel
}  // namespace OpenViBE
