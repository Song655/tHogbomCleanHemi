/// @copyright (c) 2011 CSIRO
/// Australia Telescope National Facility (ATNF)
/// Commonwealth Scientific and Industrial Research Organisation (CSIRO)
/// PO Box 76, Epping NSW 1710, Australia
/// atnf-enquiries@csiro.au
///
/// This file is part of the ASKAP software distribution.
///
/// The ASKAP software distribution is free software: you can redistribute it
/// and/or modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 2 of the License,
/// or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
///
/// @detail
///
/// @author Ben Humphreys <ben.humphreys@csiro.au>

// System includes
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstddef>
#include <cmath>
#include <sys/stat.h>
#include <omp.h>

// Local includes
#include "Parameters.h"
#include "Stopwatch.h"
#include "HogbomGolden.h"
#include "HogbomHemi.h"

#include "fitsio.h"

using namespace std;


void printerror( int status)
{
    /*****************************************************/
    /* Print out cfitsio error messages and exit program */
    /*****************************************************/

    if (status)
    {
       fits_report_error(stderr, status); /* print error report */

       exit( status );    /* terminate the program, returning error status */
    }
    return;
}

vector<float> readImage(const string& filename)
{

    fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
    int status,  nfound, anynull;
    long naxes[2], fpixel, nbuffer, npixels, ii, sizeV;

#define buffsize 1000
    float nullval, buffer[buffsize];

    status = 0;
    sizeV = 0;

    if ( fits_open_file(&fptr, filename.c_str(), READONLY, &status) )
         printerror( status );

    /* read the NAXIS1 and NAXIS2 keyword to get image size */
    if ( fits_read_keys_lng(fptr, "NAXIS", 1, 2, naxes, &nfound, &status) )
         printerror( status );

    npixels  = naxes[0] * naxes[1];         /* number of pixels in the image */
    naxis_1 = naxes[0];
    naxis_2 = naxes[1];
	
    fpixel   = 1;
    nullval  = 0;                /* don't check for null values in the image */
    vector<float> image(npixels);

    while (npixels > 0)
    {
      nbuffer = npixels;
      if (npixels > buffsize)
        nbuffer = buffsize;     /* read as many pixels as will fit in buffer */

      /* Note that even though the FITS images contains unsigned integer */
      /* pixel values (or more accurately, signed integer pixels with    */
      /* a bias of 32768),  this routine is reading the values into a    */
      /* float array.   Cfitsio automatically performs the datatype      */
      /* conversion in cases like this.                                  */

      if ( fits_read_img(fptr, TFLOAT, fpixel, nbuffer, &nullval,
                  buffer, &anynull, &status) )
           printerror( status );

      for (ii = 0; ii < nbuffer; ii++)  {
	image[sizeV++] = buffer[ii];	
      }
      npixels -= nbuffer;    /* increment remaining number of pixels */
      fpixel  += nbuffer;    /* next pixel to be read in image */
    }

    if ( fits_close_file(fptr, &status) )
         printerror( status );
/*
    struct stat results;
    if (stat(filename.c_str(), &results) != 0) {
        cerr << "Error: Could not stat " << filename << endl;
        exit(1);
    }

    vector<float> image(results.st_size / sizeof(float));
    ifstream file(filename.c_str(), ios::in | ios::binary);
    file.read(reinterpret_cast<char *>(&image[0]), results.st_size);
    file.close();
*/
    return image;
}

void writeImage(const string& filename, vector<float>& image)
{
/*
    ofstream file(filename.c_str(), ios::out | ios::binary | ios::trunc);
    file.write(reinterpret_cast<char *>(&image[0]), image.size() * sizeof(float));
    file.close();
*/
    fitsfile *infptr, *outfptr;  /* pointer to input and output FITS files */
    int status, nkeys, keypos, ii, jj;
    long  fpixel, nelements, exposure;
    char infilename[]  = "dirty.fits"; 

    long naxes[2] = { naxis_1, naxis_2 };  
    char card[FLEN_CARD];   /* standard string lengths defined in fitsioc.h */

    status = 0;         /* initialize status before calling fitsio routines */

    if ( fits_open_file(&infptr, infilename, READONLY, &status) )
         printerror( status );

    if (fits_create_file(&outfptr, filename.c_str(), &status)) /* create new FITS file */
         printerror( status );           /* call printerror if error occurs */

   /* get number of keywords */
    if ( fits_get_hdrpos(infptr, &nkeys, &keypos, &status) ) 
         printerror( status );

    /* copy all the keywords from the input to the output extension */
    for (ii = 1; ii <= nkeys; ii++)  {
        fits_read_record (infptr, ii, card, &status); 
        fits_write_record(outfptr,    card, &status); 
    }
        
    fpixel = 1;                               /* first pixel to write      */
    nelements = naxes[0] * naxes[1];          /* number of pixels to write */

    /* write the array of unsigned integers to the FITS file */
    if ( fits_write_img(outfptr, TFLOAT, fpixel, nelements, &image[0], &status) )
        printerror( status );
                

    if ( fits_close_file(outfptr, &status) )                /* close the file */
         printerror( status );           

    return;

}

size_t checkSquare(vector<float>& vec)
{
    const size_t size = vec.size();
    const size_t singleDim = sqrt(size);
    if (singleDim * singleDim != size) {
        cerr << "Error: Image is not square" << endl;
        exit(1);
    }

    return singleDim;
}

void zeroInit(vector<float>& vec)
{
    for (vector<float>::size_type i = 0; i < vec.size(); ++i) {
        vec[i] = 0.0;
    }
}

bool compare(const vector<float>& expected, const vector<float>& actual)
{
    if (expected.size() != actual.size()) {
        cout << "Fail (Vector sizes differ)" << endl;
        return false;
    }

    const size_t len = expected.size();
    for (size_t i = 0; i < len; ++i) {
        if (fabs(expected[i] - actual[i]) > 0.00001) {
            cout << "Fail (Expected " << expected[i] << " got "
                << actual[i] << " at index " << i << ")" << endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv)
{
    cout << "Reading dirty image and psf image" << endl;
    // Load dirty image and psf
    vector<float> dirty = readImage(g_dirtyFile);
    const size_t dim = checkSquare(dirty);
    vector<float> psf = readImage(g_psfFile);
    const size_t psfDim = checkSquare(psf);

    bool computeGolden = true;
    if (argc > 1 && !strstr(argv[0], "skipgolden"))
        computeGolden = false;

    // Reports some numbers
    cout << "Iterations = " << g_niters << endl;
    cout << "Image dimensions = " << dim << "x" << dim << endl;

    vector<float> goldenResidual;
    vector<float> goldenModel(dirty.size());
    if (computeGolden)
    {
        //
        // Run the golden (basic serial) version of the code
        //
        zeroInit(goldenModel);
        {
            // Now we can do the timing for the serial (Golden) CPU implementation
            cout << "+++++ Forward processing (CPU Golden) +++++" << endl;
            HogbomGolden golden;

            Stopwatch sw;
            sw.start();
            golden.deconvolve(dirty, dim, psf, psfDim, goldenModel, goldenResidual);
            const double time = sw.stop();

            // Report on timings
            cout << "    Time " << time << " (s) " << endl;
            cout << "    Time per cycle " << time / g_niters * 1000 << " (ms)" << endl;
            cout << "    Cleaning rate  " << g_niters / time << " (iterations per second)" << endl;
            cout << "Done" << endl;
        }

        // Write images out
        writeImage("residual.fits", goldenResidual);
        writeImage("model.fits", goldenModel);
    }
    else
    {
        goldenResidual = readImage("residual.fits");
        goldenModel = readImage("model.fits");
    }

    //
    // Run the Parallel version of the code
    //
    vector<float> ompResidual(dirty.size());
    vector<float> ompModel(dirty.size());
    zeroInit(ompModel);
    {
        // Now we can do the timing for the parallel implementation
        HogbomHemi hogbomHemi(psf);

        Stopwatch sw;
        sw.start();
        hogbomHemi.deconvolve(dirty, dim, psfDim, ompModel, ompResidual);
        const double time = sw.stop();

        // Report on timings
        cout << "    Time " << time << " (s) " << endl;
        cout << "    Time per cycle " << time / g_niters * 1000 << " (ms)" << endl;
        cout << "    Cleaning rate  " << g_niters / time << " (iterations per second)" << endl;
        cout << "Done" << endl;
    }

    cout << "Verifying model...";
    const bool modelDiff = compare(goldenModel, ompModel);
    if (!modelDiff) {
        return 1;
    } else {
        cout << "Pass" << endl;
    }

    cout << "Verifying residual...";
    const bool residualDiff = compare(goldenResidual, ompResidual);
    if (!residualDiff) {
        return 1;
    } else {
        cout << "Pass" << endl;
    }

    return 0;
}
