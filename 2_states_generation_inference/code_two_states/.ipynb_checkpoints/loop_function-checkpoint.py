import numpy as np

def loop_inf_partners(d_l_iter,func,**kwargs):
    key = list(d_l_iter.keys())[0]
    l = []
    for x in d_l_iter[key]:
        kwargs[key]=x
        l.append(func(kwargs["l_msa"],kwargs["s_train"],kwargs["reg"],kwargs["n_pair"],theta=kwargs["theta"],fast=kwargs["fast"]))
    return np.array(l)

def loop_on_function(n_iter,function,*args):
    l = []
    for i in range(n_iter):
        l.append(function(*args))
    return np.array(l)

def loop_sampling_parameter(n_avg,d_l_iter,func,**kwargs):
    key = list(d_l_iter.keys())[0]
    l = []
    for x in d_l_iter[key]:
        kwargs[key]=x
        l.append(loop_on_function(n_avg,func,kwargs["n_generations"],kwargs["n_mutations_branch"],kwargs["start_equi"]))
    return l

def loop_inf_partners_data(func, data, theta, *args):
    """
    call func(x,*args,theta=theta,fast=True) with x in data
    """
    l=[]
    for x in data:            
        l.append(func(x,*args, theta=theta, fast=True))      
    return np.array(l)

def loop_sampling_temperature(n_avg, l_temp, sampling, func, *args):
    l = []
    for T in l_temp:
        sampling.T = T
        l.append(loop_on_function(n_avg,func,*args))
    return l

##### Different Graph ########

def loop_sampling_temperature_wolf(n_avg, l_temp, sampling, func, *args):
    l = []
    for T in l_temp:
        sampling.T = T
        if args[-1]!=0:
            args = list(args)
            args[-1] = 5 + (T-1)*5
        l.append(loop_on_function(n_avg,func,*args))
    return l

def loop_on_list(ll,func,*args,**kwargs):
    """
    call func(args[0],x,*args[1:]) with x in ll
    """
    l = []
    args = list(args)
    for i,x in enumerate(ll):
        if "l_middle_index" in kwargs.keys():
            middle_index = kwargs["l_middle_index"][i]
            l.append(func(args[0], x, middle_index, *args[1:]))           
        else:
            l.append(func(args[0], x, *args[1:]))
    return np.array(l)


def loop_inf_partners_diff_graph(infe_partn, ll_msa, **kwargs):
    ll=[]
    for i,l_msa in enumerate(ll_msa):
        l=[]
        for msa in l_msa:
            l.append(infe_partn(msa, kwargs["s_train"], kwargs["reg"], kwargs["n_pair"], theta=kwargs["theta"], fast=kwargs["fast"], middle_index=kwargs["l_middle_index"][i])) 
        ll.append(l)
    return np.array(ll)

def loop_inf_partners_temp_mut(d_l_iter,func,**kwargs):
    key = list(d_l_iter.keys())[0]
    kwargs["l_msa"] = np.array(kwargs["l_msa"])
    lll=[]
    for it in range(kwargs["l_msa"].shape[0]):
        ll = []
        for im in range(kwargs["l_msa"].shape[1]):
            l = []
            for x in d_l_iter[key]:
                kwargs[key]=x
                l.append(func(kwargs["l_msa"][it,im],kwargs["s_train"],kwargs["reg"],kwargs["n_pair"],theta=kwargs["theta"],fast=kwargs["fast"]))
            ll.append(l)
        lll.append(ll)
    return np.array(lll)

def loop_inf_partners_temp(d_l_iter,func,**kwargs):
    key = list(d_l_iter.keys())[0]
    kwargs["l_msa"] = np.array(kwargs["l_msa"])
    ll = []
    for it in range(kwargs["l_msa"].shape[0]):
        l = []
        for x in d_l_iter[key]:
            kwargs[key]=x
            l.append(func(kwargs["l_msa"][it],kwargs["s_train"],kwargs["reg"],kwargs["n_pair"],theta=kwargs["theta"],fast=kwargs["fast"]))
        ll.append(l)
    return np.array(ll)