#include <vector>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <omp.h>
#include "pss.h"
#include "utilities.h"


#include "../lib/Eigen/Core"

using namespace Eigen;
using namespace std;

#define mMaxVelKmS	20

#define mErrReturn(msg) \
    cout << "Error: " << msg << endl; return -1


int main( int argc, char** argv )
{
    if ( argc<2 )
    {
	cout << "Run sai and oput in whole files (instead each cdp)" << endl; 
	mErrReturn("Run executable as: ./saiw config.par");
    }

    std::string parfile(argv[1]);

    IOPar par;
    par.getPar( parfile.c_str() );
    const IOPar::SetupInfo& su = *par.setupInfo();

    const int nrz = (su.t1-su.t0)/su.tstep+1;
    const int nrangles = (su.a1-su.a0)/su.astep+1;
    const int wvletsz = (su.sw[2]-su.sw[0])/su.sw[1]+1;

    VectorXd ang( nrangles );
    for ( int i=0; i<nrangles; ++i )
	ang(i) = su.a0+i*su.astep;
   
    const bool usehorconstrain = !su.horinp.empty(); 
    std::vector<int> h0, h1; 
    if ( usehorconstrain )
    {
    	if ( !Utilities::readHorizonConstrainsInMs(su.horinp.c_str(),h0,h1) )
	{
	    cout << "Error: could not read horizon file " 
		 << su.horinp.c_str() << endl; 
	    return -1;
	}
    
	if ( h0.size() != su.nrcdp )
    	{
    	    mErrReturn("Horizon size doesn't match with configure cdp size");
    	}

	const int minh = *std::min_element(h0.begin(), h0.end());
	if ( minh < su.t0 )
	{
    	    mErrReturn("Minimum horizon time is not in data time range");
	}
	const int maxh = *std::max_element(h1.begin(), h1.end());
	if ( maxh > su.t1 )
	{
    	    mErrReturn("Maximum horizon time is not in data time range");
	}
    }

    if ( su.jobdir.empty() )
    {
	mErrReturn("Please define a processing jobdir in configure file");
    }
    
    const size_t startidx = su.startcdp <= 0 ? 0 : su.startcdp-1;
    const size_t stopidx = su.stopcdp <= 0 || su.stopcdp > su.nrcdp ? 
	su.nrcdp-1 : su.stopcdp-1;

    time_t tstart = time(0);
    cout<<"-----Simulated Annealing Inversion Started--------"<<endl;
    cout<<"-----Start time: "<<ctime(&tstart)<<endl;
    cout<<"-----Total number of CDPs: "<<su.nrcdp<<endl;
    cout<<"-----Node CDP range (" <<startidx+1<<"--"<<stopidx+1<< ")"<<endl;

//OpenMP part
/*    
    if ( su.nrthreads > 0 )
    	omp_set_num_threads( su.nrthreads );
#pragma omp parallel for default(none) schedule(dynamic) shared(su,h0,h1,ang,std::cout)
*/
    for ( size_t idx=startidx; idx<=stopidx; idx++ )
    {
	//The above for loop is the main loop we need to optimize, 
	//since they are independent traces and could process one on each thread

	//Reading in the corresponding trace data for processing, each trace 
	// stores its' results to a directory start from 000001, 000002, ....
    	const int curnrdigits = Utilities::nrDigits( idx+1 );
	std::string curdir( 6-curnrdigits, '0' );
	
	std::ostringstream fullpath;
	fullpath << su.jobdir.c_str() << "/" << curdir.c_str() << idx+1;

	struct stat inf; 
	if ( stat(fullpath.str().c_str(), &inf)!=0 )
	{
    	    std::string cmd( "mkdir " ); cmd.append( fullpath.str().c_str() );
    	    system( cmd.c_str() );
	}
	else
	{
	    std::string pnm( fullpath.str().c_str() ); pnm.append("/vp.o");
	    std::string snm( fullpath.str().c_str() ); snm.append("/vs.o");
	    std::string dnm( fullpath.str().c_str() ); dnm.append("/den.o");
	    std::string pinm( fullpath.str().c_str() ); pinm.append("/pi.o");
	    if ( stat(pnm.c_str(),&inf) == 0 && stat(snm.c_str(),&inf) == 0 &&
		 stat(dnm.c_str(),&inf) == 0 && stat(pinm.c_str(),&inf) == 0 )
		continue;
	}

	const int cnrz = usehorconstrain ? (h1[idx]-h0[idx])/su.tstep+1 : nrz;
	const bool need_trim_data = cnrz < nrz;
	const int z0 = usehorconstrain ? (h0[idx]-su.t0)/su.tstep : 0;

	std::vector<float> ag;
	int sidx0 = idx*nrangles*nrz;
	int sz = nrangles*nrz;
	Utilities::fileBinaryRead<float>( ag, su.aginp.c_str(), sidx0, sz );
	MatrixXd s_data( cnrz, nrangles );
	if ( need_trim_data )
	{
	    for ( int i=0; i<nrangles; ++i )
	    {
		std::vector<double> ag1( ag.begin()+i*nrz+z0, 
			ag.begin()+i*nrz+z0+cnrz );
		s_data.col(i) = Eigen::Map<VectorXd>(ag1.data(),cnrz);
	    }
	}
	else
	{
    	    std::vector<double> ag1( ag.begin(), ag.end() );
    	    s_data = Eigen::Map<MatrixXd>(ag1.data(),nrz,nrangles);
	}

	std::vector<float> wv;
	sidx0 = idx*nrangles*wvletsz;
	sz = nrangles*wvletsz;
	Utilities::fileBinaryRead<float>( wv, su.wvinp.c_str(), sidx0, sz );
	std::vector<double> wv1( wv.begin(), wv.end() );
	MatrixXd w_data = Eigen::Map<MatrixXd>( wv1.data(), wvletsz, nrangles );

	sidx0 = idx*nrz+z0;
	BGModel bm;
	std::vector<float> bvp, bvs, bden;
	if ( !Utilities::fileBinaryRead<float>(bvp,su.vpinp.c_str(),sidx0,cnrz)
	 ||!Utilities::fileBinaryRead<float>(bvs,su.vsinp.c_str(),sidx0,cnrz)
	 ||!Utilities::fileBinaryRead<float>(bden,su.deninp.c_str(),sidx0,cnrz))
	{
	    cout << " CDP err at " << idx+1 << endl;
	}
        bm.vp = std::vector<double>(bvp.begin(),bvp.end());
    	bm.vs = std::vector<double>(bvs.begin(),bvs.end());
	bm.den = std::vector<double>(bden.begin(),bden.end());
	
	WellData wd;
	wd.cdpid = idx+1;
	wd.maxiter = su.maxitn;
	wd.errtol = su.errtol;
	wd.errthrehold = su.errthrehold;

	//Do sminulation: GPU part from here	
	ModelSimulator pss;
    	pss.simulate( s_data, w_data, ang, bm, wd );

	//Store results to its folder
	std::vector<float> mvpr, mvsr, mdenr;
	const bool saver = wd.vpr.size() > 0;
	if ( saver )
	{
    	    for ( size_t i=0; i<wd.vpr.size(); ++i )
    	    {
		mvpr.push_back( wd.vpr[i] );
		mvsr.push_back( wd.vsr[i] );
		mdenr.push_back( wd.denr[i] );
	    }

	    std::string prn( fullpath.str().c_str() ); prn.append("/vpr.o");
    	    std::string srn( fullpath.str().c_str() ); srn.append("/vsr.o");
    	    std::string drn( fullpath.str().c_str() ); drn.append("/denr.o");
    	    Utilities::writeBinaryData( prn.c_str(), mvpr.data(), cnrz );
    	    Utilities::writeBinaryData( srn.c_str(), mvsr.data(), cnrz );
    	    Utilities::writeBinaryData( drn.c_str(), mdenr.data(), cnrz );
	}

	if ( wd.vp.size()==0 )
	    continue;

	std::vector<float> mvp(cnrz,0), mvs(cnrz,0), mden(cnrz,0), mpi(cnrz,0);
	for ( size_t i=0; i<wd.vp.size(); ++i )
	{
	    mvp[i] = wd.vp[i];
	    mvs[i] = wd.vs[i];
	    mden[i] = wd.den[i];
	    mpi[i] = (wd.vp[i] - su.pic*wd.vs[i])*wd.den[i];
	}

	std::string vpnm( fullpath.str().c_str() ); vpnm.append("/vp.o");
	std::string vsnm( fullpath.str().c_str() ); vsnm.append("/vs.o");
	std::string dennm( fullpath.str().c_str() ); dennm.append("/den.o");
	std::string pinm( fullpath.str().c_str() ); pinm.append("/pi.o");
	Utilities::writeBinaryData( vpnm.c_str(), mvp.data(), cnrz );
	Utilities::writeBinaryData( vsnm.c_str(), mvs.data(), cnrz );
	Utilities::writeBinaryData( dennm.c_str(), mden.data(), cnrz );
	Utilities::writeBinaryData( pinm.c_str(), mpi.data(), cnrz );
    }

    time_t tend = time(0);
    float dlibtime = difftime(tend,tstart)/60;
    cout<<"----------SAI Processing done--------"<<endl;
    cout<<"----------Current time: "<<ctime(&tend)<<endl;
    cout<<"----------Total running minutes is: "<<dlibtime<<endl;

    return 0;
}
