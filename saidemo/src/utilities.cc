#include "utilities.h"
#include <fstream>
#include <sstream>

using namespace Eigen;

template<typename T> 
bool Utilities::fileAsciiRead( const char* filename, 
			       std::vector<std::vector<T> >& matrix )
{
    std::vector<T> singlecol;
    int issinglecol = -1;
    std::ifstream in( filename );
    if ( in.is_open() )
    {
	std::string line;
	while (!in.eof() )
	{
	    getline(in, line);
	    std::stringstream ss(line);
	    
	    std::vector<T> rowvals;
	    T currentval;

	    while( ss >>currentval )
		rowvals.push_back( currentval );

	    if ( rowvals.size()==0 )
		continue;
	    
	    if ( issinglecol==-1 )
		issinglecol = rowvals.size()==1 ? 1 : 0;
	    
	    if ( issinglecol==1 )
		singlecol.push_back( currentval );
	    else
		matrix.push_back( rowvals );
	}
	in.close();
    }

    if ( issinglecol==1 )
	matrix.push_back( singlecol );
    
    return matrix.size();
}


template<typename T> 
bool Utilities::fileBinaryRead( std::vector<T>& res, const char* fname, 
				int startidx, std::size_t sz )
{
    std::ifstream fin( fname, std::ios::binary | std::ios::in );
    if ( !fin ) 
    {
	std::cout << "Could not open file " << fname << std::endl;
	return false;
    }

    fin.seekg(0, fin.end);
    std::size_t datasz = fin.tellg()/sizeof(T);
    if ( sz<=0 ) sz = datasz;
    if ( datasz < sz )
    {
	std::cout << "Ask size " << sz << " is bigger than file size " 
	    	  << datasz << std::endl;
	return false;
    }

    if ( startidx<0 ) startidx = 0;
    fin.seekg( startidx*sizeof(T) );
    res.resize( sz, (T)0.0 );
    fin.read( reinterpret_cast<char*>(res.data()), sz*sizeof(T) );
    fin.close();

    return true;
};

template<typename T> 
bool Utilities::writeBinaryData( const char* filename,  const T* arr, 
				  std::size_t sz )
{
    std::ofstream binaryio;
    binaryio.open( filename, std::ios::binary );
    if ( !binaryio.is_open() )
	return false;

    binaryio.write( reinterpret_cast<const char*>(arr), sz * sizeof(T) );
    binaryio.close();
    return true;
}


template<typename T> 
bool Utilities::writeAsciiData( const char* fname, const T* arr, size_t sz )
{
    std::ofstream asciiio( fname );
    if ( !asciiio.is_open() )
	return false;

    for ( size_t idx=0; idx<sz; idx++ )
    	asciiio << arr[idx] << "\t";

    asciiio.close();
    return true;
}


template
bool Utilities::writeAsciiData(const char* fname,const float* arr,size_t);

template
bool Utilities::writeAsciiData(const char* fname,const double* arr,size_t);

template
bool Utilities::writeAsciiData(const char* fname,const int* arr,size_t);

template
bool Utilities::writeBinaryData(const char*,const float*,std::size_t);

template
bool Utilities::writeBinaryData(const char*,const double*,std::size_t);

template
bool Utilities::fileBinaryRead(std::vector<float>& res,const char* fname,
				int startidx,std::size_t sz);
template
bool Utilities::fileBinaryRead(std::vector<double>& re,const char* fname,
				int startidx,std::size_t sz);
template
bool Utilities::fileAsciiRead(const char* fname,
			      std::vector<std::vector<float> >& res);
template
bool Utilities::fileAsciiRead(const char* fname,
			      std::vector<std::vector<double> >& res);

bool Utilities::readHorizonConstrainsInMs( const char* filename,
					   std::vector<int>& hor0, 
   					   std::vector<int>& hor1 )
{
    std::ifstream in( filename );
    if ( !in.is_open() )
    {
	std::cout << "Error: could not open horizon file " 
	    	  << filename << std::endl;
	return false;
    }

    std::string line;
    while (!in.eof() )
    {
	getline(in, line);
	if ( line.empty() || line[0] == '#' )
	    continue;

	std::stringstream ss(line);
	std::size_t cdpnr;
	int z0, z1;

	ss >> cdpnr >> z0 >> z1;
	hor0.push_back( z0 );
	hor1.push_back( z1 );
    }

    in.close();
    return hor0.size()>0;
}


float Utilities::readCDPCorrelation( const char* filename )
{
    std::ifstream in( filename );

    float res = 0;
    if ( in.is_open() )
    {
	std::string line;
	while (!in.eof() )
	{
	    getline(in, line);
	    if ( line.empty() )
		continue;

	    std::stringstream ss(line);
	    ss >> res;
	    break;
	}
	in.close(); 
    }
    
    return res;
}


bool Utilities::readXYFile( const char* filename, std::vector<float>& xs,
						  std::vector<float>& ys )
{
    std::ifstream in( filename );
    if ( !in.is_open() )
    {
	std::cout << "Error: could not open xyfile " 
	    	  << filename << std::endl;
	return false;
    }

    std::string line;
    while (!in.eof() )
    {
	getline(in, line);
	if ( line.empty() || line[0] == '#' )
	    continue;

	std::stringstream ss(line);
	float x, y;

	ss >> x >> y;
	xs.push_back( x );
	ys.push_back( y );
    }

    in.close();
    return xs.size() > 0;
}


bool Utilities::readHorizonConstrainsInS( const char* filename,
					  std::vector<float>& hor0, 
   					  std::vector<float>& hor1 )
{
    std::ifstream in( filename );
    if ( !in.is_open() )
    {
	std::cout << "Error: could not open horizon file " 
	    	  << filename << std::endl;
	return false;
    }

    std::string line;
    while (!in.eof() )
    {
	getline(in, line);
	if ( line.empty() || line[0] == '#' )
	    continue;

	std::stringstream ss(line);
	std::size_t cdpnr;
	float z0, z1;

	ss >> cdpnr >> z0 >> z1;
	hor0.push_back( z0 );
	hor1.push_back( z1 );
    }

    in.close();

    const std::size_t nrcdp = hor0.size();
    if ( nrcdp > 0 && hor0[nrcdp-1] >10 ) //meaning hor is in ms
    {
	for ( std::size_t idx=0; idx<nrcdp; ++idx )
	{
	    hor0[idx] = hor0[idx] * 0.001;
	    hor1[idx] = hor1[idx] * 0.001;
	}
    }

    return nrcdp>0;
}


bool Utilities::readMultiChannelModel( const char* filename, int nrangles, 
				       std::size_t idx0, std::size_t idx1,
			       	       MatrixXf& ag )
{
    VectorXf tmp;
    readOneChannelModel( filename, idx0, idx1, tmp );
    int ci = 0;
    for ( int zidx=0; zidx<ag.cols(); zidx++ )
    {
    	for ( int aidx=0; aidx<ag.rows(); aidx++ )
	    ag( zidx, aidx ) = tmp[ci++];
    }

    return true;
}


bool Utilities::readOneChannelModel( const char* filename, std::size_t idx0, 
				     std::size_t idx1, VectorXf& res )
{
    if ( idx0 > idx1 )
    {
	std::cout << "Error: start index " << idx0 << " is bigger than " 
	          << idx1 << std::endl;
	return false;
    }

    std::ifstream fin( filename, std::ios::binary | std::ios::in );
    if ( !fin ) 
    {
	std::cout << "Error: could not open background file " 
	    	  << filename << std::endl;
	return false;
    }

    fin.seekg(0, fin.end);
    std::size_t datasz = fin.tellg()/sizeof(float);
    std::size_t sz = idx1 - idx0 + 1;
    if ( datasz < sz )
    {
	std::cout << "Ask size " << sz << " is bigger than file size " 
	    	  << datasz << std::endl;
	return false;
    }

    fin.seekg( idx0*sizeof(float) );
    res.resize( sz, 0.0f );
    fin.read( reinterpret_cast<char*>(res.data()), sz*sizeof(float) );
    fin.close();

    return true;
};


int Utilities::nrDigits( size_t num )
{
    int nrdigits = 0;
    size_t tmp = num;
    do
    {
	nrdigits++;
	tmp = tmp/10;
    } while ( tmp > 0 );
    
    return nrdigits;
}


template<typename T>
void Utilities::convz( const T* ref, size_t rsz, const T* wvlet, size_t wsz, 
		       std::vector<T>& s, int nzero, int nout, bool usetaper )
{
    if ( nzero==-1 )
	nzero = round( (wsz+1)/2 );
    else if ( nzero<=0 )
	nzero = 1;

    if ( nout==-1 )
	nout = rsz; 

    const int ilr = rsz-1;
    const int ilw = wsz-1;
    const int ihi = nout+nzero-1;
    s.resize( nout );
    for ( int i=nzero-1; i<ihi; ++i )
    {
	int jlow = i-ilw;
	if ( jlow < 0 ) jlow = 0;
	
	int jhigh = i;
	if ( jhigh > ilr ) jhigh = ilr;
	
	T sum = 0.;
	for ( int j=jlow; j<=jhigh; ++j )
	    sum += ref[j]*wvlet[i-j];
	
	s[i-nzero+1] = sum;
    }
}

template
void Utilities::convz( const float* ref, size_t rsz, const float* wvlet, 
	size_t wsz, std::vector<float>& s, int nzero, int nout, bool usetaper);
template
void Utilities::convz( const double* ref, size_t rsz, const double* wvlet, 
	size_t wsz, std::vector<double>& s, int nzero, int nout, bool usetaper);


template<typename T>
void::Utilities::convz( const Eigen::Matrix<T,Eigen::Dynamic,1>& r,
			const Eigen::Matrix<T,Eigen::Dynamic,1>& w,
			Eigen::Matrix<T,Eigen::Dynamic,1>& s,
			int nzero, int nout, bool usetap )
{
    if ( nzero==-1 )
	nzero = round( (w.size()+1)/2 );
    else if ( nzero<=0 )
	nzero = 1;

    if ( nout==-1 )
	nout = r.size(); 

    const int ilr = r.size()-1;
    const int ilw = w.size()-1;
    const int ihi = nout+nzero-1;
    s.resize( nout, 1 );
    for ( int i=nzero-1; i<ihi; ++i )
    {
	int jlow = i-ilw;
	if ( jlow < 0 ) jlow = 0;
	
	int jhigh = i;
	if ( jhigh > ilr ) jhigh = ilr;
	
	T sum = 0.;
	for ( int j=jlow; j<=jhigh; ++j )
	    sum += r[j]*w[i-j];
	
	s[i-nzero+1] = sum;
    }
}

template
void::Utilities::convz( const Eigen::Matrix<float,Eigen::Dynamic,1>& r,
			const Eigen::Matrix<float,Eigen::Dynamic,1>& w,
			Eigen::Matrix<float,Eigen::Dynamic,1>& output,
			int nzero, int nout, bool usetap );
template
void::Utilities::convz( const Eigen::Matrix<double,Eigen::Dynamic,1>& r,
			const Eigen::Matrix<double,Eigen::Dynamic,1>& w,
			Eigen::Matrix<double,Eigen::Dynamic,1>& output,
			int nzero, int nout, bool usetap );

void Utilities::convz( const VectorXf& r, const VectorXf& w, VectorXf& s,
		       int nzero, int nout, bool usetaper )
{
    if ( nzero==-1 )
	nzero = round( (w.size()+1)/2 );
    else if ( nzero<=0 )
	nzero = 1;

    if ( nout==-1 )
	nout = r.size(); 

    const int ilr = r.size()-1;
    const int ilw = w.size()-1;
    const int ihi = nout+nzero-1;
    s.resize( nout );
    for ( int i=nzero-1; i<ihi; ++i )
    {
	int jlow = i-ilw;
	if ( jlow < 0 ) jlow = 0;
	
	int jhigh = i;
	if ( jhigh > ilr ) jhigh = ilr;
	
	float sum = 0.;
	for ( int j=jlow; j<=jhigh; ++j )
	    sum += r[j]*w[i-j];
	
	s[i-nzero+1] = sum;
    }
}


template<typename T>
T Utilities::mean( const vectorX& x )
{ return x.col(0).mean(); }

template
float Utilities::mean(const Eigen::Matrix<float,Eigen::Dynamic,1>& x);

template
double Utilities::mean(const Eigen::Matrix<double,Eigen::Dynamic,1>& x);

template<typename T>
T Utilities::variance( const vectorX& x )
{
    T mean = Utilities::mean(x); 
    vectorX dif = x - mean * vectorX::Ones(x.size(),1);
    return dif.col(0).dot( dif.col(0) ) / dif.col(0).size();

    vectorX tmp = dif.array() * dif.array();
    return Utilities::mean( tmp ); 
}

template
float Utilities::variance(const Eigen::Matrix<float,Eigen::Dynamic,1>& x);

template
double Utilities::variance(const Eigen::Matrix<double,Eigen::Dynamic,1>& x);

template<typename T>
T Utilities::covariance( const vectorX& x, const vectorX& y )
{
    T meanx = Utilities::mean<T>(x);
    T meany = Utilities::mean<T>(y);

    vectorX dx = x-meanx*vectorX::Ones(x.size(),1);
    vectorX dy = y-meany*vectorX::Ones(y.size(),1);
    return dx.col(0).dot(dy.col(0)) / dx.col(0).size();

    vectorX tmp3 = dx.array()*dy.array();
    return Utilities::mean<T>(tmp3);
}


double Utilities::pearsonCorrelationD( const VectorXd& x, const VectorXd& y )
{
    const double meanx = x.mean();
    const double meany = y.mean();
    VectorXd dx = x-meanx*VectorXd::Ones(x.size());
    VectorXd dy = y-meany*VectorXd::Ones(y.size());
    const double dm = dx.dot(dx) * dy.dot(dy);
    return dm <= 1e-18 ? 0 : dx.dot(dy) / sqrt(dm);
}


template<typename T>
T Utilities::pearsonCorrelation( const vectorX& x, const vectorX& y )
{
    /*
    T meanx = Utilities::mean<T>(x);
    T meany = Utilities::mean<T>(y);

    vectorX dx = x - meanx*vectorX::Ones(x.size(),1);
    vectorX dy = y - meany*vectorX::Ones(y.size(),1);
    T sqx2 = dx.col(0).dot( dx.col(0) );
    T sqy2 = dy.col(0).dot( dy.col(0) );
    return dx.col(0).dot(dy.col(0)) / sqrt(sqx2 * sqy2);
    */
    T vx = Utilities::variance(x);
    T vy = Utilities::variance(y);
    T cv = Utilities::covariance(x,y);
    T dm = vx * vy;
    return dm <= 1e-18 ? 0 : sqrt((cv*cv)/dm);
}

template
float Utilities::pearsonCorrelation(
	const Eigen::Matrix<float,Eigen::Dynamic,1>& x,
	const Eigen::Matrix<float,Eigen::Dynamic,1>& y);
template
double Utilities::pearsonCorrelation(
	const Eigen::Matrix<double,Eigen::Dynamic,1>&,
	const Eigen::Matrix<double,Eigen::Dynamic,1>&);

double Utilities::norm1( const Eigen::VectorXd& v )
{
    return v.cwiseAbs().sum();
}

float Utilities::norm1( const Eigen::VectorXf& v )
{
    return v.cwiseAbs().sum();

    float sum = 0.;
    for ( int idx=0; idx<v.size(); ++idx )
	sum += std::abs(v[idx]);
    return sum;
}


double Utilities::normInf( const Eigen::VectorXd& v )
{ return v.cwiseAbs().maxCoeff(); }


float Utilities::normInf( const Eigen::VectorXf& v )
{
    return v.cwiseAbs().maxCoeff();

    float max = -1.;
    float value;
    for ( int idx=0; idx<v.size(); ++idx )
    {
	value = std::abs(v[idx]);
	if ( value > max )
	    max = value;
    }
    return max;
}


double Utilities::norm2( const Eigen::VectorXd& v )
{ return sqrt(v.cwiseAbs2().sum()); }


float Utilities::norm2( const Eigen::VectorXf& v )
{
    return sqrt(v.cwiseAbs2().sum());

    float sum = 0.;
    for ( int idx=0; idx<v.size(); ++idx )
	sum += v[idx]*v[idx];
    return sqrt(sum);
}


IOPar::IOPar()
{
    suinf_ = new SetupInfo();
}

IOPar::~IOPar()
{
    delete suinf_;
}


bool IOPar:: readHosts(const char* hostfile, std::vector<std::string>& lst )
{
    if ( !hostfile )
	return false;

    std::ifstream file( hostfile );
    if ( !file.is_open() ) 
    {
	std::cout << "Error: could not open host file " 
	    	  << hostfile << std::endl;
	return 0;
    }

    std::string line;
    while ( std::getline(file,line) )
    {
	if ( line.empty() || line[0] == '#' )
	    continue;
	    
	std::string key;
	
	std::istringstream iss(line);
	iss >> key;
	lst.push_back( key );
    }

    file.close();
    return true;
}


size_t IOPar::getNrCDP( const char* configfile )
{
    if ( !configfile )
	return 0;

    std::ifstream file( configfile );
    if ( !file.is_open() ) 
    {
	std::cout << "Error: could not open configure file " 
	    	  << configfile << std::endl;
	return 0;
    }

    std::string line;
    while ( std::getline(file,line) )
    {
	if ( line.empty() || line[0] == '#' )
	    continue;
	    
	std::string key, dum, valuestr;
    	float value;
	
	std::istringstream iss(line);
	iss >> key >> dum >> value;
	if ( !key.compare("nrcdp") )
	{
	    file.close();
	    return (std::size_t)value;
	}
    }

    file.close();
    return 0;
}


void IOPar::getPar( const char* configfile )
{
    if ( !configfile )
	return;

    std::ifstream file( configfile );
    if ( !file.is_open() ) 
    {
	std::cout << "Error: could not open configure file " 
	    	  << configfile << std::endl;
	return;
    }

    std::string line;
    while ( std::getline(file,line) )
    {
	if ( line.empty() || line[0] == '#' )
	    continue;
	    
	std::string key, dum, valuestr;
    	double value;
	
	std::istringstream iss(line);
	iss >> key >> dum >> value;
	if ( !key.compare("nrthreads") )
	    suinf_->nrthreads = (std::size_t)value;
	else if ( !key.compare("is2d") )
	    suinf_->is2d = (int)value==1;
	else if ( !key.compare("imprefl") )
	    suinf_->imprefl = (int)value==1;
	else if ( !key.compare("nrcdp") )
	    suinf_->nrcdp = (std::size_t)value;
	else if ( !key.compare("startcdp") )
	    suinf_->startcdp = (std::size_t)value;
	else if ( !key.compare("stopcdp") )
	    suinf_->stopcdp = (std::size_t)value;
	else if ( !key.compare("t0") )
	    suinf_->t0 = (int)value;
	else if ( !key.compare("t1") )
	    suinf_->t1 = (int)value;
	else if ( !key.compare("tstep") )
	    suinf_->tstep = (int)value;
	else if ( !key.compare("a0") )
	    suinf_->a0 = (int)value;
	else if ( !key.compare("a1") )
	    suinf_->a1 = (int)value;
	else if ( !key.compare("astep") )
	    suinf_->astep = (int)value;
	else if ( !key.compare("pic") )
	    suinf_->pic = value;
	else if ( !key.compare("sw") )
	{
	    suinf_->sw[0] = (int)value;
	    iss >> dum >> value;
	    suinf_->sw[1] = (int)value;
	    iss >> dum >> value;
	    suinf_->sw[2] = (int)value;
	}
	else if ( !key.compare("wvtype") )
	{
	    suinf_->wvtype = (int)value;
	}
	else if ( !key.compare("maxitn") )
	    suinf_->maxitn = (int)value;
	else if ( !key.compare("errtol") )
	    suinf_->errtol = value;
	else if ( !key.compare("errthrehold") )
	    suinf_->errthrehold = value;
	else if ( !key.compare("nsmooth") )
	    suinf_->nsmooth = (int)value;
	else if ( !key.compare("solver") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->solver = tmp;
	}
	else if ( !key.compare("hosts") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->hosts = tmp;
	}
	else if ( !key.compare("aginp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->aginp = tmp;
	}
	else if ( !key.compare("wvinp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->wvinp = tmp;
	}
	else if ( !key.compare("vpinp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->vpinp = tmp;
	}
	else if ( !key.compare("vsinp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->vsinp = tmp;
	}
	else if ( !key.compare("deninp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->deninp = tmp;
	}
	else if ( !key.compare("horinp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->horinp = tmp;
	}
	else if ( !key.compare("vpoutp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->vpoutp = tmp;
	}
	else if ( !key.compare("vsoutp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->vsoutp = tmp;
	}
	else if ( !key.compare("denoutp") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->denoutp = tmp;
	}
	else if ( !key.compare("jobdir") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->jobdir = tmp;
	}
	else if ( !key.compare("cljobdir") )
	    suinf_->cljobdir = (int)value==1;
	else if ( !key.compare("vpr") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->vpr = tmp;
	}
	else if ( !key.compare("vsr") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->vsr = tmp;
	}
	else if ( !key.compare("denr") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->denr = tmp;
	}
	else if ( !key.compare("xyfile") )
	{
	    std::string tmp;
	    std::istringstream iss1(line);
    	    iss1 >> key >> dum >> tmp;
	    suinf_->xyfile = tmp;
	}
    }

    file.close();
}


IOPar::SetupInfo::SetupInfo()
{
    nrthreads = -1;
    is2d = false;
    imprefl = false;
    nrcdp = 0;
    startcdp = 0;
    stopcdp = 0;
    t0 = 0;
    t1 = 0;
    tstep = 1;
    a0 = 0;
    a1 = 0;
    astep = 1;
    pic = 1.4;
    sw[0] = sw[1] = 0;
    sw[2] = 1;
    maxitn = 1000;
    errtol = 1e-8;
    errthrehold = 0.5;
    solver = "bpi";
    hosts = "";
    aginp = "";
    wvinp = "";
    wvtype = 0;
    vpinp = "";
    vsinp = "";
    deninp = "";
    horinp = "";
    vpoutp = "";
    vsoutp = "";
    denoutp = "";
    jobdir = "";
    cljobdir = false;
    vpr = "";
    vsr = "";
    denr = "";
    xyfile = "";
}


