#pragma once

#include <vector>
#include <fstream>
#include <iostream>

#include <../lib/Eigen/Dense>

#define mAbs(x) ((x) > 0. ? x : -(x))
#define mMax(x,y)       ((x)>(y) ? (x) : (y))
#define mMin(x,y)       ((x)<(y) ? (x) : (y))

#define DBL_MAX	1e+28

using namespace Eigen;
using namespace std;


/* 
   Solve a L1-regularized least squares problem:
   min ||A*x-b||^2 + lambda*sum|x_i| 

Parameters:
    A : MxN Matrix input data Columns correspond to features
    b : M vector outcome
   lambda : >0, regularization parameter
    x0 : initial guess of solution  
    At : NxM transpose of A
    M : number of examples (rows) of A
    N : number of features (columns) of A
    tar_gap : relative target duality gap
    eta : PCG termination 
    pcgmaxi : maximum PCG iterations

Returns:
    x : array_like classifier.
    history : matrix history data. 
    	columns represent (truncated) Newton iterations; 
    	rows represent the following:
    	- 1st row gap
    	- 2nd row primal objective
    	- 3rd row dual objective
    	- 4th row step size
    	- 5th row pcg status flag (-1 = error, 1 = failed, 0 = success)

void runL1LSTest()
{
    MatrixXd A(3,4);
    A << 1., 0., 0., 0.5,
         0., 1., 0.2, 0.3,
	 0., 0.1, 1., 0.2;
    VectorXd x0(4);
    x0 << 1., 0., 1., 0.;
    double lambda = 0.01;
    double rel_tol = 0.01;
    VectorXd y = A*x0;

    VectorXd x;

    const bool isok = L1LS::solve(A,y,x, lambda,0,rel_tol);
    if ( !isok )
	cout << "Running L1LS error happended " << endl;
}
 */

class L1LS
{
public:
				L1LS() {}

    static bool			solve(const Eigen::MatrixXd& A,
	    			      const Eigen::VectorXd& b,
				      Eigen::VectorXd& x,
				      std::string& err,
				      double lambda, 
				      Eigen::VectorXd* x0=0,
				      double tar_gap=1e-3,
				      double eta=1e-3,
				      int pcgmaxi=5000);
    static void			pcgSolver(const Eigen::VectorXd& b,
    					  double pcgtol, int pcgmaxi,
					  const Eigen::MatrixXd& A,
					  const Eigen::MatrixXd& At,
					  const Eigen::VectorXd& d1,
					  const Eigen::VectorXd& d2,
					  const Eigen::VectorXd& p1,
					  const Eigen::VectorXd& p2,
					  const Eigen::VectorXd& p3,
					  Eigen::VectorXd& dxu,
					  int& pflg,int& piter);

    static double		norm1(const Eigen::VectorXd&);
    static double		norm2(const Eigen::VectorXd&);
    static double		normInf(const Eigen::VectorXd&);

protected:    
    static void			pcgAX(const Eigen::VectorXd& x,
	    			      const Eigen::MatrixXd& A,
	    			      const Eigen::MatrixXd& At,
	    			      const Eigen::VectorXd& d1,
	    			      const Eigen::VectorXd& d2,
	    			      const Eigen::VectorXd& p1,
	    			      const Eigen::VectorXd& p2,
	    			      const Eigen::VectorXd& p3,
				      Eigen::VectorXd& res);
    static void			pcgIX(const Eigen::VectorXd& x,
	    			      const Eigen::VectorXd& p1,
	    			      const Eigen::VectorXd& p2,
	    			      const Eigen::VectorXd& p3,
				      Eigen::VectorXd& res);
};


#define mErrRtn(msg) \
    err = std::string(msg); return false

bool L1LS::solve( const MatrixXd& A, const VectorXd& y, 
		  VectorXd& x, std::string& err, double lambda, 
		  VectorXd* x0, double tar_gap, double eta, int pcgmaxi )
{
    if ( y.size()==0 )
    {
	mErrRtn( "Defined y vector is required!" );
    }

    const int m = A.rows();
    const int n = A.cols();
    if ( m<0 || n<0 ) 
    {
	mErrRtn( "Matrix size has to be positive!" );
    }

    const MatrixXd& At = A.transpose();

    //Interior Point Method(IPM) Parameters
    const int MU = 2;  		//updating parameter of t 
    const int MAX_NT_ITER = 400;//IPM max Newton iteration

    //LINE SEARCH PARAMETERS
    const double ALPHA = 0.01;	//minimum fraction of decrease in the objective
    const double BETA = 0.5;		//stepsize decrease factor
    const int MAX_LS_ITER = 100;//maximum backtracking line search iteration

    if ( x0 )
	x = *x0;
    else
    	x = VectorXd::Zero(n);

    double t = mMin( mMax(1,1/lambda), 2*n/1e-3 );
    VectorXd u = VectorXd::Ones(n), newu(n), newx; 
    VectorXd f(2*n), newf(2*n); 
    f << x-u, -x-u;

    double dobj = -DBL_MAX;
    double s   = DBL_MAX; 
    int pitr  = 0 ; 
    int pflg  = 0 ;

    VectorXd dxu = VectorXd::Zero(2*n); 
    VectorXd diagxtx = VectorXd::Constant(n,2.0); 

    VectorXd z = VectorXd::Zero(m), newz(m); 
    VectorXd nu = VectorXd::Zero(m); 

    int lsiter = 0;
    int ntiter;
    for ( ntiter=0; ntiter<MAX_NT_ITER; ++ntiter )
    {
	z = A*x-y;

	//Duality gap calculation	
	nu = 2*z;

	const double maxAnu = normInf(At*nu);
	if ( maxAnu > lambda )
	    nu = (lambda/maxAnu)*nu;

	const double pobj  =  z.dot(z)+lambda*norm1(x);
	double tmp = -0.25*nu.dot(nu)-nu.dot(y);
	dobj  =  mMax(tmp,dobj);
	const double gap   =  pobj - dobj;

	//STOPPING CRITERION
	if ( gap/dobj < tar_gap )
	    return true;

	if ( s >= 0.5 )
	    t = mMax( mMin(2*n*MU/gap, MU*t), t );

	//CALCULATE NEWTON STEP
	VectorXd q1 = (u+x).cwiseInverse();
	VectorXd q2 = (u-x).cwiseInverse();
	VectorXd q12 = q1.array()*q1.array();
	VectorXd q22 = q2.array()*q2.array();
	VectorXd d1 = (q12+q22)/t;
	VectorXd d2 = (q12-q22)/t;

	//Calculate gradient
	VectorXd g1 = At*(z*2)-(q1-q2)/t;
	VectorXd g2 = lambda*VectorXd::Ones(n)-(q1+q2)/t;
	VectorXd gradphi( g1.size()+g2.size() );
	gradphi << g1, g2;

	//calculate vectors to be used in the preconditioner
	VectorXd prb = diagxtx+d1;
	VectorXd prs = prb.array()*d1.array()-d2.array()*d2.array();

	//set pcg tolerance (relative)
	const double normg = norm2(gradphi);
	double pcgtol  = mMin(1e-1,eta*gap/mMin(1,normg));

	if ( ntiter != 0 && pitr == 0 ) 
	    pcgtol = pcgtol*0.1;

	VectorXd iprs = prs.cwiseInverse();
	VectorXd p1 = d1.array()*iprs.array();
	VectorXd p2 = d2.array()*iprs.array();
	VectorXd p3 = prb.array()*iprs.array();
	pcgSolver(-gradphi, pcgtol, pcgmaxi,
	       A, At, d1, d2, p1, p2, p3, dxu, pflg, pitr );

	if ( pflg==1 ) 
	    pitr = pcgmaxi; 

	VectorXd dx  = dxu.head(n);
	VectorXd du  = dxu.tail(n);

	//BACKTRACKING LINE SEARCH
	double phi = z.dot(z)+lambda*u.sum()-(-f).array().log().sum()/t;
	s = 1.0;
	const double gdx = gradphi.dot(dxu);
	for ( lsiter=1; lsiter <=MAX_LS_ITER; ++ lsiter )
	{
	    newx = x+s*dx; 
	    newu = u+s*du;
	    newf << newx-newu, -newx-newu;
	    if ( newf.maxCoeff()<0 )
	    {
		newz   =  A*newx-y;
		double newphi =  newz.dot(newz)+lambda*newu.sum() -
		    (-newf).array().log().sum()/t;
		if ( newphi-phi <= ALPHA*s*gdx )
		    break;
	    }

	    s = BETA*s;
	}

	if ( lsiter==MAX_LS_ITER ) 
	    break; //exit by BLS

	x = newx; u = newu; f = newf;
    }

    //ABNORMAL TERMINATION (FALL THROUGH)
    if ( lsiter == MAX_LS_ITER )
    {
	mErrRtn("MAX_LS_ITER exceeded in BLS");

    }
    else if ( ntiter == MAX_NT_ITER )
    {
	mErrRtn("MAX_NT_ITER exceeded");
    }
    
    return true;
}

void L1LS::pcgSolver( const Eigen::VectorXd& b, double tol, int maxiter,
		      const Eigen::MatrixXd& A,
		      const Eigen::MatrixXd& At,
		      const Eigen::VectorXd& d1,
		      const Eigen::VectorXd& d2,
		      const Eigen::VectorXd& p1,
		      const Eigen::VectorXd& p2,
		      const Eigen::VectorXd& p3,
		      VectorXd& dxu, int& pflag, int& piter )
{
    VectorXd x = dxu.size()==0 ? VectorXd::Zero(b.size()) : dxu;
    maxiter += 2;

    if ( tol<=0. )
	tol = 1e-6;

    bool matrix_positive_definite = true;
    VectorXd p = VectorXd::Zero( b.size() );
    double oldtau = 1;

    VectorXd ax;
    pcgAX(x,A,At,d1,d2,p1,p2,p3,ax);
    VectorXd r = b-ax;
    double b_norm = norm2(b);
    VectorXd resvec(maxiter);
    resvec(0) = norm2(r);
    double alpha = 1;
    piter = 2;
    while ( resvec(piter-2) > tol*b_norm && piter<maxiter )
    {
	VectorXd y;
	pcgIX(r,p1,p2,p3,y);
	VectorXd z = y;
	double tau = z.dot(r);
	double beta = tau / oldtau;
	oldtau = tau;
	p = z+beta*p;
	VectorXd w;
	pcgAX(p,A,At,d1,d2,p1,p2,p3,w);
	alpha = tau/p.dot(w);
	if ( alpha<0 )
	    matrix_positive_definite = false;
	x = x+alpha*p;
	r = r-alpha*w;
	resvec(piter-1) = norm2(r);
	piter++;
    }

    pflag = 0;
    if ( piter>maxiter-2 )
	pflag = 1;
    if ( !matrix_positive_definite )
	pflag = 3;
    piter -= 2;
    dxu = x;
}


double L1LS::norm1( const Eigen::VectorXd& v )
{
    double sum = 0.;
    for ( int idx=0; idx<v.size(); ++idx )
	sum += std::abs(v[idx]);
    return sum;
}


double L1LS::normInf( const Eigen::VectorXd& v )
{
    double max = -1.;
    double value;
    for ( int idx=0; idx<v.size(); ++idx )
    {
	value = std::abs(v[idx]);
	if ( value > max )
	    max = value;
    }
    return max;
}


double L1LS::norm2( const Eigen::VectorXd& v )
{
    double sum = 0.;
    for ( int idx=0; idx<v.size(); ++idx )
	sum += v[idx]*v[idx];
    return sqrt(sum);
}


void L1LS::pcgAX( const VectorXd& x, const MatrixXd& A, const MatrixXd& At, 
		  const VectorXd& d1, const VectorXd& d2, const VectorXd& p1, 
		  const VectorXd& p2, const VectorXd& p3, VectorXd& res	)
{
    const int n = x.size()/2;
    VectorXd x1 = x.head(n);
    VectorXd x2 = x.tail(n);
    VectorXd tmp = d1.array()*x1.array()+d2.array()*x2.array();
    VectorXd y1 = 2*(At*(A*x1))+tmp; 
    VectorXd y2 = d2.array()*x1.array()+d1.array()*x2.array();
    res.resize(y1.size()+y1.size());
    res << y1, y2;
}


void L1LS::pcgIX( const VectorXd& x, const VectorXd& p1, 
		  const VectorXd& p2, const VectorXd& p3, VectorXd& res	)
{
    const int n = x.size()/2;
    VectorXd x1 = x.head(n);
    VectorXd x2 = x.tail(n);
    VectorXd y1 = p1.array()*x1.array()-p2.array()*x2.array(); 
    VectorXd y2 = -p2.array()*x1.array()+p3.array()*x2.array();
    res.resize(y1.size()+y1.size());
    res << y1, y2;
}



