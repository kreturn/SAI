#pragma once

#include <vector>
#include "../lib/Eigen/Core"


using namespace Eigen;

struct BGModel 
{
    std::vector<double>	vp;
    std::vector<double>	vs;
    std::vector<double>	den;
};


struct WellData 
{
    std::vector<double>	vp;
    std::vector<double>	vs;
    std::vector<double>	den;
    std::vector<double>	vpr;
    std::vector<double>	vsr;
    std::vector<double>	denr;
    int			cdpid;
    int			maxiter;
    double		errtol;
    double		errthrehold;
    int			iterdone;
    double		err;
    double		corr; //corelation between seis and synthetic
};


struct InvPar
{
    MatrixXd	waveformmatrix_compact;
    VectorXd	angdata;
    BGModel	bmodel;
    VectorXi	id_index;
    int		nsmooth;
    double	pi_c;
};
    

class ModelSimulator
{
public:    
    			ModelSimulator()	{}

    void		simulate(const MatrixXd& s_data,const MatrixXd& w_data,
				 const VectorXd& ang,const BGModel& bmodel,
				 WellData& res,
				 VectorXi id_flag=VectorXi::Zero(1),
				 VectorXi target_flag = VectorXi::Zero(1),
				 int nsmooth = 20,
				 double pi_c = 1.4,
				 bool display = true);
    static double 	modelOptFun(const VectorXd& x,const InvPar& d);

protected:

    MatrixXd		toeplitz(const VectorXd& wltmatrix,int ncol);
    MatrixXd		extractCols(const MatrixXd& D,
	    			    const std::vector<int>& index);
    MatrixXd		extractRows(const MatrixXd& D,
	    			    const std::vector<int>& index);
    VectorXd		extractItems(const VectorXd& D,
	    			     const std::vector<int>& index);
    static VectorXd	velFromReflectivity(const VectorXd&r,double v0);
    static void		modelForward(const VectorXd& x,
	    				const InvPar& d,WellData& wd);
    static std::vector<double>	movemean(const std::vector<double>& a,int len);
};





