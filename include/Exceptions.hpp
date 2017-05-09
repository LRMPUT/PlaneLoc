/*
 * Exceptions.h
 *
 *  Created on: 14 cze 2016
 *      Author: jachu
 */

#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <exception>
#include <string>
#include <sstream>
#include <typeinfo>
#ifdef __linux__
#include <execinfo.h>
#endif

#define PLANE_EXCEPTION(msg) plane_exception(__FILE__, __LINE__, std::string(msg))

class plane_exception : public std::exception
{
public:
	plane_exception(const std::string& ifile,
				const size_t iline,
				const std::string& imsg = std::string(),
				const std::string& exName = "pgm_exception")
		: msg(imsg),
		  file(ifile),
		  line(iline)
	{
		createWhatMsg(exName);
	}

	plane_exception(const std::string& ifile,
				const size_t iline,
				const std::exception& srcEx,
				const std::string& exName = "pgm_exception")
		: file(ifile),
		  line(iline)
	{
		msg = "wrapped exception class '" + std::string(typeid(srcEx).name()) + "': " + std::string(srcEx.what());
		createWhatMsg(exName);
	}
	virtual ~plane_exception() throw() {};

	const char* what() const throw() {
		return whatMsg.c_str();
	}

	const char* get_file() const throw() {
		return file.c_str();
	}

	size_t get_line() const throw() {
		return line;
	}

	virtual const char* get_message() {
		return msg.c_str();
	}

private:
	std::string msg;
	std::string file;
	size_t line;

	std::string whatMsg;

	void createWhatMsg(const std::string& ex_name){
		std::stringstream ss;
		ss << "Exception '" << ex_name << "' thrown in file '" << file << "' line '" << line << "' with message:\n" << get_message();

#ifdef __linux__
		void *buffer[100];
		char **strings;

		int nptrs = backtrace(buffer, 100);

		strings = backtrace_symbols(buffer, nptrs);
		if (strings != nullptr) {
			ss << std::endl << std::endl << "Stack trace:" << std::endl;
			for (int j = 0; j < nptrs; j++)
				ss << strings[j] << std::endl;
			free(strings);
		}
#endif
		whatMsg = ss.str();
	}
};


#endif /* EXCEPTIONS_H_ */
