#include("/home/ubuntu/my_data/data8/cfos_app/timetable/CYCLOPS-2.0-main/CYCLOPS_2_0_Template.jl")


using DataFrames, Statistics, StatsBase, LinearAlgebra, MultivariateStats, PyPlot, Distributed, Random, CSV, Revise, Distributions, Dates, MultipleTesting
using CUDA
CUDA.set_runtime_version!("12.0")

# out_postf = ARGS[1] * "_" * ARGS[2] * "_" * ARGS[3]
print(ARGS[2]* "\n")
print(ARGS[3]* "\n")
print(ARGS[4]* "\n") 

base_path = joinpath(pwd(), "", "") # define the base path in which the data file, the seed gene list, and the sample collection time file are located
data_path = joinpath(base_path, "data") # path for data folder in base folder
# path_to_cyclops = joinpath(base_path, "CYCLOPS.jl") # path to CYCLOPS.jl
# print(path_to_cyclops)
output_path = joinpath(base_path, "results") # real training run output folder in base folder

expression_data = CSV.read(joinpath(data_path, "all_data_cr.csv"), DataFrame) # load data file in data folder
print(expression_data[1:5,1:5])
seed_genes = ["LHA","AMd","MA","RPF","PRNr","CA1","AAA","IAM","SubG","VISal1","Mmm","AV","CM","mtg","APr","MD","IMD","SOCm","SH","Xi","AD","APN","POST","Pa4","CA2","SNc","ACAd2/3","ECT2/3","PoT","CL","RL","Mml","VPLpc","RE","III","RSPv2/3","PERI1","PERI6b","RSPv5","SMT","RN","RSPv1","SSp-ll1","AMv","ECT5","SSp-un6b","VPMpc","SGN","RH","EW","LD","NPC","MGm","PH","OV","PCN","LP","FL","fi","SCdw","PAR","pc","csc","SCO","PR","DT","tspc","RT","VISpl5","SPA","AVPV","PPT","PIL","NB","ENTm2","PVT","act","ENTm3","VISpl2/3","AVP","AIv1","LPO","LDT","dhc","sm","SUB","MPT","LA","sctd","SCH","NTB","PP","ACAd1","bsc","MPO","AHN","NOT","VMPO","VISpor5","LGd-sh","VISpl1","RSPd2/3","SPFm","moV","MEPO","LGv","RPA","DG-po","das","RCH","OP","NDB","SLC","LING","PFL","LGd-ip","int","amc","ENTm5","VLPO","VISpm5","FN","och","SCsg","SCzo","RO","PPY","SAG","LGd-co","ccs","VISpor2/3","DG-sg","VISpl4","MARN","IO","VISp5","NOD","PGRNl","IP","CUN","y","PPN","SOCl","ENTm1","ICB","ACAv1","PVpo","CUL4, 5","UVU","VeCB","RSPd1","V3","VISl5","VISpor1","tb","COPY","DG-mo","ENTm6","ll","SCop","scwm","or","POR","IG","IA","ccb","PVa","BAC","SSs1","mcp","RM","FOTU","LAV","ENTl1","sup","VISpm1","LRNm","DN","SUV","V4r","DEC","VISl1","MV","arb","PYR","IVn","chpl","PVi","VISp4","PRP","hbc","SSs6b","VISl4","VISp1","SPIV","AP","gVIIn","NLL","bic","VII","PERI2/3","SUT","ICc","SPVI","ICe","ICd","SSp-m1","PVH","vVIIIn","VISpm4","sptV","SPVC","PC5","PRM","VISl2/3","SO","GRN","MDRNd","opt","PST","PVHd","sctv","pyd","KF","VCO","ECU","ACVII","LIN","V","MDRNv","I5","PSV","SSp-m5","GU6b","P5","MOp5","SSp-ul6a","VISp2/3","PARN","NR","IRN","SSp-m6b","MOp6a","SSs6a","SSp-m6a","GU6a","SSs5","XII","SPVO","SSp-m2/3","ts","SSp-n6a","nst","Acs5","ISN","AUDv5","SNr","PMv","NTS","DMX","PVp","AUDp5","Pa5","AUDp6a","AUDpo4","AUDpo5","AUDp4","GU2/3","IntG","AUDd6a","AUDpo6a","AUDd5","VISpm6b","SCiw","PD","AId6a","RSPd6b","AUDd4","ACAv6a","SFO","RSPagl6a","VISal6a","OT","AUDv2/3","AUDv4","GU4","IPC","SSp-n2/3","SF","FRP6a","VISpm6a","Mmme","RSPagl6b","AUDp2/3","SSp-n4","AUDpo2/3","IPI","SSp-tr4","ORBl6a","RSPd6a","ORBvl6a","INC","FRP2/3","AId6b","VISa6a","IPL","CLA","RSPv6a","mp","vtd","ORBl6b","MS","VISam6b","ACAd6a","CS","VISam6a","VISa4","AIv6b","NI","SSp-bfd5","SLD","FRP5","VISrl6a","ORBl2/3","AId2/3","B","ORBl5","ORBvl5","AId5","VISal5","IPA","VISrl5","DR","VISa5","RPO","ORBvl6b","LH","LSc","ORBvl2/3","LSv","ACAv5","ORBm5","TTv","SSp-bfd4","LSr","IPRL","AIv5","IPR","ND","ORBm2/3","IF","AON","GPi","NLOT3","SSp-tr2/3","SSp-un4","SSp-ll4","Su3","SSp-bfd2/3","VTN","PN","PL2/3","VISam5","FF","BLAv","AT","dscp","IAD","GPe","VISa2/3","SSp-un2/3","VISrl2/3","SSp-ll2/3","SSp-bfd1","CLI","ACAv2/3","VM","mfb","PF","TTd","VISal2/3","VISam2/3","ILA2/3","MOB","Eth","dtd","ACAd5","vhc","mlf","AIv2/3","PeF","VISrl1","DTN","BLAa","DP"]# print(seed_genes)
sample_ids_with_collection_times = Array{String, 1}(["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","96","97","98","99","100","101","102","103","104","105","106","107","108","109","110","111","112","113","114","115","116","117","118","119","120","121","122","123","124","125","126","127","128","129","130","131","132","133","134","135","136","137","138","139","140","141","142","143","144"]) # sample ids for which collection times exist
sample_collection_times = Array{Float64, 1}([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 1.0471975511965976, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 3.141592653589793, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989, 5.235987755982989]) # colletion times for sample ids

#
if ((length(sample_ids_with_collection_times)+length(sample_collection_times))>0) && (length(sample_ids_with_collection_times) != length(sample_collection_times))
    error("ATTENTION REQUIRED! Number of sample ids provided (\'sample_ids_with_collection_times\') must match number of collection times (\'sample_collection_times\').")
end

# make changes to training parameters, if required. Below are the defaults for the current version of cyclops.
training_parameters = Dict(:regex_cont => r".*_C",			# What is the regex match for continuous covariates in the data file
:regex_disc => r".*_D",							# What is the regex match for discontinuous covariates in the data file

:blunt_percent => 0.975, 						# What is the percentile cutoff below (lower) and above (upper) which values are capped

:seed_min_CV => parse(Float64, ARGS[4]),#0.25, 							# The minimum coefficient of variation a gene of interest may have to be included in eigen gene transformation
:seed_max_CV => parse(Float64, ARGS[5]),#0.5, 							# The maximum coefficient of a variation a gene of interest may have to be included in eigen gene transformation
:seed_mth_Gene => parse(Int, ARGS[1]), 						# The minimum mean a gene of interest may have to be included in eigen gene transformation

:norm_gene_level => true, 						# Does mean normalization occur at the seed gene level
:norm_disc => false, 							# Does batch mean normalization occur at the seed gene level
:norm_disc_cov => 1, 							# Which discontinuous covariate is used to mean normalize seed level data

:eigen_reg => true, 							# Does regression again a covariate occur at the eigen gene level
:eigen_reg_disc_cov => 1, 		 				# Which discontinous covariate is used for regression
:eigen_reg_exclude => false,					# Are eigen genes with r squared greater than cutoff removed from final eigen data output
:eigen_reg_r_squared_cutoff => 0.6,				# This cutoff is used to determine whether an eigen gene is excluded from final eigen data used for training
:eigen_reg_remove_correct => false,				# Is the first eigen gene removed (true --> default) or it's contributed variance of the first eigne gene corrected by batch regression (false)

:eigen_first_var => false, 						# Is a captured variance cutoff on the first eigen gene used
:eigen_first_var_cutoff => 0.85, 				# Cutoff used on captured variance of first eigen gene

:eigen_total_var => 0.85, 						# Minimum amount of variance required to be captured by included dimensions of eigen gene data
:eigen_contr_var => 0.05, 						# Minimum amount of variance required to be captured by a single dimension of eigen gene data
:eigen_var_override => true,					# Is the minimum amount of contributed variance ignored
:eigen_max => parse(Int, ARGS[2]), 								# Maximum number of dimensions allowed to be kept in eigen gene data

:out_covariates => true, 						# Are covariates included in eigen gene data
:out_use_disc_cov => true,						# Are discontinuous covariates included in eigen gene data
:out_all_disc_cov => true, 						# Are all discontinuous covariates included if included in eigen gene data
:out_disc_cov => 1,								# Which discontinuous covariates are included at the bottom of the eigen gene data, if not all discontinuous covariates
:out_use_cont_cov => false,						# Are continuous covariates included in eigen data
:out_all_cont_cov => true,						# Are all continuous covariates included in eigen gene data
:out_use_norm_cont_cov => false,				# Are continuous covariates Normalized
:out_all_norm_cont_cov => true,					# Are all continuous covariates normalized
:out_cont_cov => 1,								# Which continuous covariates are included at the bottom of the eigen gene data, if not all continuous covariates, or which continuous covariates are normalized if not all
:out_norm_cont_cov => 1,						# Which continuous covariates are normalized if not all continuous covariates are included, and only specific ones are included

:init_scale_change => true,						# Are scales changed
:init_scale_1 => false,							# Are all scales initialized such that the model sees them all as having scale 1
                                                # Or they'll be initilized halfway between 1 and their regression estimate.

:train_n_models => parse(Int, ARGS[3]), 							# How many models are being trained
:train_μA => 0.001, 							# Learning rate of ADAM optimizer
:train_β => (0.9, 0.999), 						# β parameter for ADAM optimizer
:train_min_steps => 1500, 						# Minimum number of training steps per model
:train_max_steps => 2050, 						# Maximum number of training steps per model
:train_μA_scale_lim => 1000, 					# Factor used to divide learning rate to establish smallest the learning rate may shrink to
:train_circular => false,						# Train symmetrically
:train_collection_times => true,						# Train using known times
:train_collection_time_balance => 1.0,					# How is the true time loss rescaled
# :train_sample_id => sample_ids_with_collection_times,
# :train_sample_phase => sample_collection_times,

:cosine_shift_iterations => 192,				# How many different shifts are tried to find the ideal shift
:cosine_covariate_offset => true,				# Are offsets calculated by covariates

:align_p_cutoff => 0.05,						# When aligning the acrophases, what genes are included according to the specified p-cutoff
:align_base => "radians",						# What is the base of the list (:align_acrophases or :align_phases)? "radians" or "hours"
:align_disc => false,							# Is a discontinuous covariate used to align (true or false)
:align_disc_cov => 1,							# Which discontinuous covariate is used to choose samples to separately align (is an integer)
:align_other_covariates => false,				# Are other covariates included
:align_batch_only => false,
# :align_samples => sample_ids_with_collection_times,
# :align_phases => sample_collection_times,
# :align_genes => Array{String, 1},				# A string array of genes used to align CYCLOPS fit output. Goes together with :align_acrophases
# :align_acrophases => Array{<: Number, 1}, 	# A number array of acrophases for each gene used to align CYCLOPS fit output. Goes together with :align_genes

:X_Val_k => 10,									# How many folds used in cross validation.
:X_Val_omit_size => 0.1,						# What is the fraction of samples left out per fold

:plot_use_o_cov => true,
:plot_correct_batches => true,
:plot_disc => false,
:plot_disc_cov => 1,
:plot_separate => false,
:plot_color => ["b", "orange", "g", "r", "m", "y", "k"],
:plot_only_color => true,
:plot_p_cutoff => 0.05)


# Distributed.addprocs(length(Sys.cpu_info()))
Distributed.addprocs(parse(Int, ARGS[6]))
@everywhere base_path = joinpath(pwd(), "", "")
@everywhere path_to_cyclops = joinpath(base_path, "CYCLOPS.jl") # path to CYCLOPS.jl
@everywhere include(path_to_cyclops)
# @everywhere arg1 = ARGS[1] 
# @everywhere arg2 = ARGS[2] 
# @everywhere arg3 = ARGS[3]
# @everywhere out_postf = "results_"*arg1 * "_" * arg2 * "_" * arg3
@everywhere output_path = joinpath(base_path,  "results")

# real training run
training_parameters[:align_genes] = CYCLOPS.human_homologue_gene_symbol[CYCLOPS.human_homologue_gene_symbol .!= "RORC"]
training_parameters[:align_acrophases] = CYCLOPS.mouse_acrophases[CYCLOPS.human_homologue_gene_symbol .!= "RORC"]
eigendata, modeloutputs, correlations, bestmodel, parameters = CYCLOPS.Fit(expression_data, seed_genes, training_parameters)
CYCLOPS.Align(expression_data, modeloutputs, correlations, bestmodel, parameters, output_path)
 