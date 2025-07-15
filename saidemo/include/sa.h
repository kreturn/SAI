#pragma once

#include <iostream>
#include <../lib/Eigen/Core>

#define mUdf	1e+29
#define mIsUdf(v) (fabs(v)>=mUdf || -fabs(v)<=-mUdf)


using namespace std;
using namespace Eigen;
    

template <typename F>
class SearchOperator 
{
public:

    SearchOperator( const F& f_, const VectorXd& start_, 
	    	       const VectorXd& direction_ ) 
	: f(f_), start(start_), direction(direction_)
	, matrix_r(0), scalar_r(0) {}

    SearchOperator( const F& f_, const VectorXd& start_, 
	    	       const VectorXd& direction_, VectorXd& r ) 
	: f(f_), start(start_), direction(direction_), matrix_r(&r), scalar_r(0)
    {}
 
    SearchOperator( const F& f_, const VectorXd& start_, 
	    	       const VectorXd& direction_, double& r ) 
	: f(f_), start(start_), direction(direction_), matrix_r(0), scalar_r(&r)
    {}

    double operator()(const double& x) const
    {
	return get_value(f(start + x*direction));
    }

private:

    double get_value (const double& r) const
    {
	if (scalar_r)
	    *scalar_r = r;
	
	return r;
    }

    double get_value (const VectorXd& r) const
    {
	if ( matrix_r )
	    *matrix_r = r;

	return r.dot(direction);
    }

    const F& f;
    const VectorXd& start;
    const VectorXd& direction;
    VectorXd* matrix_r;
    double* scalar_r;
};

template <typename F>
const SearchOperator<F> MakeSearchOperator( 
	const F& f, const VectorXd& start, const VectorXd& direction ) 
{ return SearchOperator<F>(f,start,direction); }


template <typename F>
const SearchOperator<F> MakeSearchOperator( const F& f, 
	const VectorXd& start, const VectorXd& direction, double& f_out )
{ return SearchOperator<F>(f,start,direction, f_out); }


template <typename F>
const SearchOperator<F> MakeSearchOperator( const F& f, 
	const VectorXd& start, const VectorXd& direction, VectorXd& grad_out )
{ return SearchOperator<F>(f,start,direction,grad_out); }

class AlgoStopOpt
{
public:
    explicit AlgoStopOpt(double mindelta=1e-7) 
	: _verbose(false)
	, _been_used(false)
	, _min_delta(mindelta)
	, _max_iter(0)
	, _cur_iter(0)
	, _prev_funct_value(0)
	, _funct_threshold(0.5)  
    {}

    AlgoStopOpt(double mindelta,unsigned long max_iter,double thr)
	: _verbose(false)
	, _been_used(false)
	, _min_delta(mindelta)
	, _max_iter(max_iter)
	, _cur_iter(0)
	, _prev_funct_value(0) 
	, _funct_threshold(thr)  
    {
	//Make sure min_delta >= 0 && max_iter > 0,
    }

    void turnOnVerbose()
    { _verbose = true; }

    AlgoStopOpt& be_verbose()
    {
	_verbose = true;
	return *this;
    }

    bool should_continue_search( const VectorXd& x, double funct_value, 
	    			 const VectorXd& grad ) 
    {
	if (_verbose)
	{
	    cout << "iteration: " << _cur_iter << "   objective: " 
		<< funct_value << endl;
	}

	++_cur_iter;
	if ( _been_used )
	{
	    if ( _max_iter != 0 && _cur_iter > _max_iter )
		return false;

	    if (funct_value < _funct_threshold && //TODO: manual set 
		std::abs(funct_value - _prev_funct_value) < _min_delta)
		return false;
	}
	
	_been_used = true;
	_prev_funct_value = funct_value;
	return true;
    }

private:
  
    bool	_verbose;
    bool 	_been_used;
    double 	_min_delta;
    unsigned long _max_iter;
    unsigned long _cur_iter;
    double 	_prev_funct_value;
    double	_funct_threshold;
};


class AlgoSearchOpt
{
public:
    
AlgoSearchOpt()
   : been_used(false)
   , been_used_twice(false) 
   {}

double	get_wolfe_rho() const	{ return 0.01; }
double	get_wolfe_sigma() const	{ return 0.9; }
unsigned long get_max_line_search_iterations() const { return 100; }

const VectorXd& get_next_direction( const VectorXd& x, double funval,
	    			    const VectorXd& funct_derivative )
{
    if ( !been_used)
    {
	been_used = true;
	H = MatrixXd::Identity(x.size(),x.size());
    }
    else
    {
	delta = (x-prev_x); 
	gamma = funct_derivative-prev_derivative;
	
	double dg = delta.dot(gamma);
	if ( !been_used_twice )
	{
	    double gg = gamma.transpose()*gamma;
	    if (std::abs(gg) > std::numeric_limits<double>::epsilon())
	    {
		double temp = dg/gg;
		if ( temp<0.01 ) 
		    temp = 0.01;
		else if ( temp>100 )
		    temp = 100;
		H = temp*MatrixXd::Identity(x.size(),x.size());
		been_used_twice = true;
	    }
	}

	Hg = H*gamma;
	gH = (gamma.transpose()*H).transpose();
	double gHg = gamma.transpose()*H*gamma;
	if ( gHg < std::numeric_limits<double>::infinity() && 
	     dg < std::numeric_limits<double>::infinity() && dg!=0 )
	{
	    H += (1 + gHg/dg)*delta*delta.transpose()/(dg) - 
		(delta*gH.transpose() + Hg*delta.transpose())/(dg);
	}
	else
	{
	    H = MatrixXd::Identity(H.rows(),H.rows());
	    been_used_twice = false;
	}
    }
    
    prev_x = x;
    prev_direction = -H*funct_derivative;
    prev_derivative = funct_derivative;
    return prev_direction;
}

private:
    bool 	been_used;
    bool 	been_used_twice;
    VectorXd 	prev_x;
    VectorXd 	prev_derivative;
    VectorXd 	prev_direction;
    MatrixXd 	H;
    VectorXd 	delta; 
    VectorXd 	gamma;
    VectorXd 	Hg;
    VectorXd 	gH;
};



class Optim
{
public:  

    struct ConstrainFunctionObject
    {
	ConstrainFunctionObject( Optim& opt_, 
		const VectorXd& x_lower_, const VectorXd& x_upper_ ) 
	    : opt(opt_),x_lower(x_lower_), x_upper(x_upper_) {}

	double operator() ( const VectorXd& x ) const
	{
	    VectorXd xd = x;
	    for ( int i=0; i<x.size(); ++i )
	    {
		if ( xd[i]<x_lower[i] )
		    xd[i] = x_lower[i];
		else if ( xd[i]>x_upper[i] )
		    xd[i] = x_upper[i];
	    }

	    return opt.functionValue(xd); 
	}

	Optim& 		opt;
	const VectorXd& x_lower;
	const VectorXd& x_upper;
    };


    			Optim()	{}
    virtual double	functionValue(const VectorXd& x)	{ return 0; }
    virtual void	functionGrad(const VectorXd& x,VectorXd& grad)	{}

    
    double	executeConstrained(AlgoSearchOpt& searchopt,
	    			  AlgoStopOpt& stopopt,
	  			  VectorXd& x0, int& finaliter,
	  			  const VectorXd& x_lower,
	  			  const VectorXd& x_upper,
	  			  double gap_eps=1e-8);
protected:   

template<typename F>
double backtrackingLSearch( const F& f,	double f0, 
	double d0, double alpha, double rho, unsigned long max_iter )
{
    if ( rho<=0 || rho>=1 )
    {
	cout<<"Ensure rho is between (0, 1) " << rho;
	return mUdf;
    }

    if ( max_iter<=0 )
	max_iter = 300;

    //make sure alpha is opposite the direction of the gradient
    if ( (d0 > 0 && alpha > 0) || (d0 < 0 && alpha < 0) )
	alpha *= -1;

    bool have_prev_alpha = false;
    double prev_alpha = 0;
    double prev_val = 0;
    unsigned long iter = 0;
    while ( true )
    {
	++iter;
	const double val = f(alpha);
	if ( val <= f0 + alpha*rho*d0 || iter >= max_iter )
	{
	    return alpha;
	}
	else
	{
	    double step, tmp;
	    if ( !have_prev_alpha )
	    {
		if ( d0 < 0 )
		{
		    tmp = poly_min_extrap(f0, d0, val);
		    step = alpha*put_in_range(0.1,0.9,tmp);
		} 
		else
		{
		    tmp = poly_min_extrap(f0, -d0, val);
		    step = alpha*put_in_range(0.1,0.9,tmp); 
		}
		have_prev_alpha = true;
	    }
	    else
	    {
		if ( d0 < 0 )
		{
		    tmp = poly_min_extrap(f0, d0, alpha, val, 
			    prev_alpha, prev_val);
		    step = put_in_range(0.1*alpha,0.9*alpha, tmp); 
		}
		else
		{
		    tmp = -poly_min_extrap(f0, -d0, -alpha, val, 
			    -prev_alpha, prev_val);
		    step = put_in_range(0.1*alpha,0.9*alpha, tmp); 
		}
	    }

	    prev_alpha = alpha;
	    prev_val = val;
	    alpha = step;
	}
    }
}


VectorXd zeroBounded(double eps,VectorXd vect,
	const VectorXd& x,const VectorXd& gradient,
	const VectorXd& x_lower,const VectorXd& x_upper)
{
    for ( long i = 0; i < gradient.size(); ++i )
    {
	const double tol = eps*std::abs(x(i));
	if ( x_lower(i)+tol >= x(i) && gradient(i) > 0 )
	    vect(i) = 0;
	else if ( x_upper(i)-tol <= x(i) && gradient(i) < 0 )
	    vect(i) = 0;
    }
    return vect;
}


VectorXd gapBounded(double eps,VectorXd vect,
		    const VectorXd& x,const VectorXd& gradient,
		    const VectorXd& x_lower,const VectorXd& x_upper)
{
    for ( long i = 0; i < gradient.size(); ++i )
    {
	const double tol = eps*std::abs(x(i));
	if ( x_lower(i)+tol >= x(i) && gradient(i) > 0 )
	    vect(i) = x_lower(i)-x(i);
	else if ( x_upper(i)-tol <= x(i) && gradient(i) < 0 )
	    vect(i) = x_upper(i)-x(i);
    }
    return vect;
}

double put_in_range(double a,double b,double val)
{
    if ( a<b )
    {
	if ( val<a )
	    return a;
	else if ( val>b )
	    return b;
    }
    else
    {
	if ( val<b )
	    return b;
	else if ( val>a )
	    return a;
    }

    return val;
}

double poly_min_extrap(double f0,double d0,double f1)
{
    const double temp = 2*(f1 - f0 - d0);
    if ( std::abs(temp) <= d0*std::numeric_limits<double>::epsilon() )
	return 0.5;

    const double alpha = -d0/temp;
    return put_in_range(0,1,alpha);
}


double poly_min_extrap(double f0,double d0,double f1,
		      double d1,double limit=1)
{
    const double n = 3*(f1 - f0) - 2*d0 - d1;
    const double e = d0 + d1 - 2*(f1 - f0);

    // find the minimum of the derivative of the polynomial
    double temp = std::max(n*n - 3*e*d0,0.0);
    if ( temp < 0 )
	return 0.5;

    temp = std::sqrt(temp);
    if ( std::abs(e) <= std::numeric_limits<double>::epsilon() )
	return 0.5;

    // figure out the two possible min values
    double x1 = (temp - n)/(3*e);
    double x2 = -(temp + n)/(3*e);

    // compute the value of the interpolating polynomial at these two points
    double y1 = f0 + d0*x1 + n*x1*x1 + e*x1*x1*x1;
    double y2 = f0 + d0*x2 + n*x2*x2 + e*x2*x2*x2;

    // pick the best point
    double x = y1 < y2 ? x1 : x2;

    return put_in_range(0,limit,x);
}


double poly_min_extrap( double f0, double d0, double x1,
			double f_x1, double x2, double f_x2 )
{
    if ( x1<=0 || x1>=x2 )
    {
	cout << "Invalid inputs were given to this function." << endl;
	cout << "x1: " << x1 << "    x2: " << x2 << endl;
	return mUdf;
    }
    
    Matrix<double,2,2> m;
    Matrix<double,2,1> v;
    
    const double aa2 = x2*x2;
    const double aa1 = x1*x1;
    m << aa2, -aa1,
      -aa2*x2, aa1*x1;   
    v << f_x1-f0-d0*x1,
         f_x2-f0-d0*x2;

    double temp = aa2*aa1*(x1-x2);
    
    // just take a guess if this happens
    if ( temp == 0. || std::fpclassify(temp) == FP_SUBNORMAL )
	return x1/2.0;

    Matrix<double,2,1> temp2 = m*v/temp;
    const double a = temp2(0);
    const double b = temp2(1);
    temp = b*b - 3*a*d0;
    if ( temp < 0 || a == 0 )
    {
	if ( f0 < f_x2 )
	    return 0;
	else
	    return x2;
    }
    temp = (-b + std::sqrt(temp))/(3*a);
    return put_in_range(0, x2, temp);
}
};


double Optim::executeConstrained( AlgoSearchOpt& searchopt,
	AlgoStopOpt& stopopt, VectorXd& x, int& finaliter, 
	const VectorXd& x_lower, const VectorXd& x_upper, double gap_eps )
{
    finaliter = 0;
    if ( x.size()!=x_lower.size() || x.size()!=x_upper.size() )
    {
	cout << "\n\tUsing executeConstrained(): "
	     << "\n\t The inputs to must be equal size vectors."
	     << "\n\t x.size():               " << x.size()
	     << "\n\t x_lower.size():         " << x_lower.size()
	     << "\n\t x_upper.size():         " << x_upper.size()
	     << endl;
	return mUdf;
    }

    VectorXd dif = x_upper-x_lower;
    const double mind = dif.minCoeff();
    if ( mind<0 )
    {
	cout << "\n\tUsing executeConstrained(): "
	     << "\n\t You have to supply proper box constraints."
	     << "\n\r min(x_upper-x_lower): " << mind << endl;
	return mUdf;
    }
    
    VectorXd g;
    double f_value = functionValue(x);
    functionGrad(x,g);
    
    if ( mIsUdf(f_value) )
    {
	//cout << "Error: infinite output" << endl;
	return mUdf;
    }
    
    VectorXd s, tmp;
    double last_alpha = 1;
    int iter = 0;
    while( stopopt.should_continue_search(x,f_value,g) )
    {
	tmp = zeroBounded(gap_eps, g, x, g, x_lower, x_upper);
	s = searchopt.get_next_direction( x, f_value, tmp );
	s = gapBounded(gap_eps,s,x,g,x_lower,x_upper);

	ConstrainFunctionObject cf(*this, x_lower,x_upper); 
	double alpha = backtrackingLSearch(
		MakeSearchOperator( cf,x, s, f_value),
    		f_value, g.dot(s), last_alpha, 
		searchopt.get_wolfe_rho(), 
    		searchopt.get_max_line_search_iterations() );
	if ( alpha == last_alpha )
	    last_alpha = std::min(last_alpha*10,1.0);
	else
	    last_alpha = alpha;
	
	x = x + alpha*s;
	if ( x_lower.size()==x.size() )
	{
    	    for ( int i=0; i<x.size(); ++i )
	    {
		if ( x[i]<x_lower[i] )
		    x[i] = x_lower[i];
		else if ( x[i]>x_upper[i] )
		    x[i] = x_upper[i];
	    }
	}
	functionGrad(x,g);
	
	if ( mIsUdf(f_value) )
	{
	    //cout << "Error: infinite output" << endl;
	    return mUdf;
	}
	
	++iter;
    }
	
    //cout <<"total_iter: "<< iter << " Err: " << f_value << endl;
    finaliter = iter;    
    return f_value;
}



    





