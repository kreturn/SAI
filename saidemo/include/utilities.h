#include <fstream>
#include <iostream>
#include <vector>

#include <../lib/Eigen/Dense>

using Eigen::MatrixXf; 
using Eigen::VectorXf; 

#define vectorX Eigen::Matrix<T,Eigen::Dynamic,1>

class Utilities
{
public:    

    template<typename T> 
    static bool			fileBinaryRead(std::vector<T>& res,
	    				       const char* fname,
					       int startidx=0,std::size_t sz=0);
    template<typename T> 
    static bool			fileAsciiRead(const char* fname,
	    				      std::vector<std::vector<T> >&);
    template<typename T>
    static bool			writeBinaryData(const char* filename,
						const T* arr,std::size_t sz);
    template<typename T>
    static bool			writeAsciiData(const char* filename,
					       const T* arr,std::size_t sz);
    static float		readCDPCorrelation(const char* filename);
    static bool			readXYFile(const char* filename, 
	    				   std::vector<float>& x,
					   std::vector<float>& y);
    static bool			readHorizonConstrainsInS(const char* filename,
					std::vector<float>& hor0,
					std::vector<float>& hor1);
    static bool			readHorizonConstrainsInMs(const char* filename,
					std::vector<int>& hor0,
					std::vector<int>& hor1);
    static bool			readOneChannelModel(const char* filename,
	    				    std::size_t idx0,std::size_t idx1,
					    Eigen::VectorXf& data);
					    //std::vector<float>& data);
    static bool			readMultiChannelModel(const char* filename,
	    				 int nrangles,
					 std::size_t idx0,std::size_t idx1,
					 Eigen::MatrixXf& ag);
    static int			nrDigits(size_t num);
    static void			convz(const Eigen::VectorXf& ref,
	    			      const Eigen::VectorXf& wvlet,
				      Eigen::VectorXf& output,
    				      int nzero=-1, 
				      //default=round((length(wavelet)+1)/2)
				      int nout=-1, //default=length(ref)
				      bool usetaper=false);
    template<typename T> 
    static void			convz(
				    const Eigen::Matrix<T,Eigen::Dynamic,1>& r,
				    const Eigen::Matrix<T,Eigen::Dynamic,1>& w,
				    Eigen::Matrix<T,Eigen::Dynamic,1>& output,
    				    int nzeo=-1,int nout=-1,bool usetap=false);
    template<typename T>
    static T			mean(const vectorX& a);
    template<typename T>
    static T			variance(const vectorX& a);
    template<typename T>
    static T			covariance(const vectorX& a,const vectorX& b);

    template<typename T>
    static T			pearsonCorrelation(const vectorX& a,
	    					   const vectorX& b);
    static double		pearsonCorrelationD(const Eigen::VectorXd& x,
	    					    const Eigen::VectorXd& y);
    template<typename T> 
    static void			convz(const T* ref,size_t rsz,
	    			      const T* wvlet,size_t wsz,
				      std::vector<T>& output,
    				      int nzero=-1, 
				      int nout=-1, 
				      bool usetaper=false);

    static float		norm1(const Eigen::VectorXf&);
    static float		norm2(const Eigen::VectorXf&);
    static float		normInf(const Eigen::VectorXf&);
    static double		norm1(const Eigen::VectorXd&);
    static double		norm2(const Eigen::VectorXd&);
    static double		normInf(const Eigen::VectorXd&);
};


class IOPar
{
public:    
    			IOPar();
			~IOPar();

    struct SetupInfo
    {
			SetupInfo();
	int		nrthreads;		
	bool		is2d;
	size_t		nrcdp;
	int		t0; 	//in ms
	int		t1; 	//in ms
	int		tstep; 	//in ms
	int		a0;	//in degree
	int		a1;	//in degree
	int		astep;	//in degree
	float		pic;	//C for PI calculation
	int		sw[3];	//wavelet info: start, step, stop
	int		wvtype; // 0=volume 1= single wavelet

	int		maxitn;
	double		errtol;
	double		errthrehold;
	int		nsmooth;
	std::string	solver; //bpi, ...
	size_t		startcdp;
	size_t		stopcdp;

	std::string	hosts;	
	std::string	aginp;	
	std::string	wvinp;	
	std::string	vpinp;	
	std::string	vsinp;	
	std::string	deninp;	
	std::string	horinp;	
	std::string	vpoutp;	
	std::string	vsoutp;	
	std::string	denoutp;	
	std::string	jobdir;
	bool		cljobdir;

	std::string	vpr;	
	std::string	vsr;	
	std::string	denr;

	std::string	xyfile;		//for 2D line only
	bool		imprefl;	//import reflectivites of vp, vs, den	
    };
    SetupInfo*		setupInfo()	{ return suinf_; }
    void		getPar(const char* configfile);   
    size_t		getNrCDP(const char* configfile); 
    static bool		readHosts(const char* nm,std::vector<std::string>& ls); 

protected:    

    SetupInfo*		suinf_;
};

