/*
 * Copyright (C) 2019 - 2020 by the emerging fields initiative 'Novel Biopolymer
 * Hydrogels for Understanding Complex Soft Tissue Biomechanics' of the FAU
 *
 * This file is part of the EFI library.
 *
 * The efi library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Author: Stefan Kaessmair
 */



#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_GNUPLOT_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_GNUPLOT_H_

// stl headers
#include <string>
#include <iostream>

// efi headers
#include <efi/base/logstream.h>

namespace efi {

class GnuplotStream
{
public:

    /// Constructor.
	GnuplotStream (bool active = true);

	/// Destructor.
	~GnuplotStream ();

	/// Create a xy-plot for given $x$- and $y$-data.
	/// @param[in] gnuplot Sink we strem to.
	/// @param[in] x @p vector of $x$-data
    /// @param[in] y @p vector of $y$-data
	/// @param[in] options Additional options for the gnuplot plot command. The
	/// gnuplot syntax is: <tt>plot '-' using 1:2 options</tt>.
	template<class Number, class OtherNumber>
	static
	void
	plot (GnuplotStream& gnuplot,
	      const std::vector<Number> x,
	      const std::vector<OtherNumber> y,
	      const std::string options = "w lines lt rgb 'red' notitle");

private:

    /// The stream operator is a firend of @p GnuplotStream. It can handle all
    /// types of input as long the operator
    /// <tt>operator<<(std::ostringstream&,const Streamable&)</tt> is defined
    /// (if the content of @p input is valid gnuplot code is another thing).
    template<class Streamable>
    friend
    GnuplotStream&
    operator<< (GnuplotStream& gnuplot, const Streamable &input);

    /// Flag indicating whether the stream is active or not.
	bool active;

	/// Pipe to gnuplot.
	FILE *gnuplotpipe;
};



//---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//



inline
GnuplotStream::GnuplotStream (bool active)
    :
    active (active),
#if defined(__linux__) || defined (__APPLE__)
    gnuplotpipe (active? popen("gnuplot -persist","w") : nullptr)
#else
    gnuplotpipe (nullptr)
#endif
{
    if (active && !gnuplotpipe)
    {
        efilog(Verbosity::normal) << "Gnuplot could not be opened. Note that "
                                     "this feature is not supported on Windows "
                                     "systems. Ignoring everything streamed to "
                                     "this object."
                                  << std::endl;
        this->active = false;
    }
}



inline
GnuplotStream::~GnuplotStream ()
{
    if (gnuplotpipe)
    {
        fprintf(gnuplotpipe,"exit\n");
        pclose(gnuplotpipe);
    }
}



template<class Number, class OtherNumber>
inline
void
GnuplotStream::
plot (GnuplotStream& gnuplot,
      const std::vector<Number> x,
      const std::vector<OtherNumber> y,
      const std::string options)
{
    if (!gnuplot.active) return;

    AssertDimension(x.size(),y.size());

    // Plotting a single point does not make sense and only causes
    // gnuplot to print an error message. To avoid this, nothing
    // happens if only a single data point is provided.
    if(x.size() > 1)
    {
        gnuplot << "plot '-' using 1:2 " << options << '\n';

        for (unsigned int i = 0; i < x.size(); ++i)
            gnuplot << x.at(i) << ' ' << y.at(i) << "\n";
        gnuplot << "\ne\n";
    }
}



template<class Streamable>
inline
GnuplotStream&
operator<< (GnuplotStream& gnuplot, const Streamable &input) {

    if (!gnuplot.active) return gnuplot;

    std::ostringstream oss;
    oss << input;

    fprintf(gnuplot.gnuplotpipe,"%s", oss.str().c_str());
    fflush(gnuplot.gnuplotpipe);

    return gnuplot;
}

}//close efi


#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_GNUPLOT_H_ */
