import numpy as np # numerics for matrices
import sys # for exiting
import copy # for deepcopying
from scipy import linalg
from numba import jit
import functools 
from numba import jit,njit
import re
from multiprocessing import Pool
import scipy
from scipy import integrate
import time

complex=np.complex128
class tb_model(object):
    def __init__(self,k_dim,r_dim,lat=None,orb=None,per=None,atom_list=None,atom_position=None,nspin=1):
        if type(k_dim).__name__!='int':
            raise Exception("\n\nArgument dim_k not an integer")
        if k_dim<0 or k_dim>4:
            raise Exception("\n\nArgument dim_k out of range. Must be between 0 and 4.")
        self._dim_k=k_dim
        if type(r_dim).__name__!='int':
            raise Exception("\n\nArgument dim_r not an integer")
        if r_dim<0 or r_dim>4:
            raise Exception("\n\nArgument dim_r out of range. Must be between 0 and 4.")
        self._dim_r=r_dim
        if lat is 'unit' or lat is None:
            self._lat=np.identity(dim_r,float)
        elif type(lat).__name__ not in ['list','ndarray']:
            raise Exception("\n\nArgument lat is not a list.")
        else:
            self._lat=np.array(lat,dtype=float)
            if self._lat.shape!=(r_dim,r_dim):
                raise Exception("\n\nWrong lat array dimensions")
        if r_dim>0:
            if np.abs(np.linalg.det(self._lat))<1.0E-6:
                raise Exception("\n\nLattice vectors length/area/volume too close to zero, or zero.") 
            if np.linalg.det(self._lat)<0.0:
                raise Exception("\n\nLattice vectors need to form right handed system.")
        if orb is None:
            self._norb=1
            self._orb=np.zeros((1,r_dim))
            print(" Orbital positions not specified. I will assume a single orbital at the origin.")
        elif type(orb).__name__=='int':
            self._norb=orb
            self._orb=np.zeros((orb,r_dim))
        elif type(orb).__name__ not in ['list','ndarray']:
            raise Exception("\n\nArgument orb is not a list or an integer")
        else:
            self._orb=np.array(orb,dtype=float)
            if len(self._orb.shape)!=2:
                raise Exception("\n\nWrong orb array rank")
            self._norb=self._orb.shape[0] # number of orbitals
            if self._orb.shape[1]!=r_dim:
                raise Exception("\n\nWrong orb array dimensions")
        # choose which self._dim_k out of self._dim_r dimensions are
        # to be considered periodic.
        if per==None:
            # by default first _dim_k dimensions are periodic
            self._per=list(range(self._dim_k))
        else:
            if len(per)!=self._dim_k:
                raise Exception("\n\nWrong choice of periodic/infinite direction!")
            # store which directions are the periodic ones
            self._per=per
        # remember number of spin components
        if nspin not in [1,2]:
            raise Exception("\n\nWrong value of nspin, must be 1 or 2!")
        self._nspin=nspin
        # by default, assume model did not come from w90 object and that
        # position operator is diagonal
        self._assume_position_operator_diagonal=True
        self._nsta=self._norb*self._nspin
        self._ham=np.zeros((1,self._nsta,self._nsta),dtype=complex)
        self._hamR=np.zeros((1,self._dim_r),dtype=int)
        if atom_list is None:
            if atom_position is not None:
                raise Exception("Wrong, you should input the atom_list")
            else:
                self._atom=[1]
                self._atom_position=[self._orb[0]]
                for a in range(1,self._norb):
                    if np.any(np.linalg.norm(self._orb[a]-self._atom_position,axis=1)<1e-1):
                        index=np.argwhere(np.linalg.norm(self._orb[a]-self._atom_position,axis=1)<1e-1)
                        self._atom[-1]+=1
                    else:
                        self._atom.append(1)
                        self._atom_position.append(self._orb[a])
            self._atom_position=np.array(self._atom_position)
            self._atom=np.array(self._atom,dtype=int)
            self._natom=int(len(self._atom))
        else:
            self._atom=np.array(atom_list,dtype=int)
            self._natom=len(atom_list)
            if atom_position is None:
                raise Exception("Wrong, you should input the atom_position")
            else:
                if len(atom_position)!=self._natom:
                    raise Exception("Wrong, the atom_list's length must equal to atom_position")
                else:
                    atom_position=np.array(atom_position)
                    for atom in atom_position:
                        if np.sum(np.linalg.norm(atom-atom_position,axis=1)<1e-5)>1:
                            raise Exception("Wrong, have two atom position locals too short")
                    self._atom_position=np.array(atom_position)
        self._rmatrix=np.zeros((1,self._dim_r,self._nsta,self._nsta),dtype=complex)
        for i in range(self._dim_r):
            if self._nspin==2:
                orb=np.append(self._orb,self._orb,axis=0)
            self._rmatrix[0,i]=np.diag(orb[:,i])
                    

    def set_hop(self,tmp,ind,ind_R,mode='set',conjugate_set=True):
        ind=np.array(ind,dtype=int)
        ind_R=np.array(ind_R,dtype=int)
        if np.any(ind<0) or np.any(ind>=self._norb):
            raise Exception("\n\nIndex ind out of scope.")
        if len(ind)!=2:
            raise Exception("\n\n length of ind need to be 2")
        if len(ind_R)!=self._dim_r:
            raise Exception("\n\n Wrong, the length of ind_R must equal to dim_r")
        if np.all(ind_R==0) and ind[0]==ind[1]:
            if type(tmp).__name__=='complex' and tmp.imag!=0:
                raise Exception("\n\n Wrong, the diagonal of [0,0,0] hamiltonian must be real")
            elif type(tmp).__name__ in ['list','ndarray']:
                tmp=np.array(tmp)
                if tmp.shape[0]!=tmp.shape[1]:
                    raise Exception("\n\n Wrong, you must use pauli matrix")
                for i in range(len(tmp)):
                    if type(tmp[i,i]).__name__=='complex' and tmp[i,i].imag!=0:
                        raise Exception("\n\n Wrong, the diagonal of [0,0,0] hamiltonian must be real")
        useham=np.zeros((self._norb,self._norb),dtype=complex)
        if self._nspin==2:
            useham[ind[0],ind[1]]=1
            if type(tmp).__name__ in ['list','ndarray']:
                useham=np.kron(tmp,useham)
            else:
                useham=np.kron(tmp*np.identity(2),useham)
        else:
            if type(tmp).__name__ in ['list','ndarray']:
                raise Exception("\n\n Wrong, using the pauli matrix must take spin=2")
            useham[ind[0],ind[1]]=tmp
        if np.all(ind_R==0) and ind[0]!=ind[1] and conjugate_set:
            useham+=useham.T.conjugate()
        T1=np.all(self._hamR==ind_R,axis=1)
        T2=np.all(self._hamR==-ind_R,axis=1)
        if mode.lower()=='add':
            if np.any(T1): 
                index=np.argwhere(T1)[[0]]
                self._ham[index]+=useham
            elif np.any(T2):
                ind_R*=-1
                index=np.argwhere(T2)[[0]]
                self._ham[index]+=useham.transpose().conjugate()
            else:
               self._hamR=np.append(self._hamR,[ind_R],axis=0)
               self._ham=np.append(self._ham,[useham],axis=0)
        elif mode.lower()=='set':
            if np.any(T1):
                index=np.argwhere(T1)[0,0]
                a=np.sum(self._ham[index]*useham)
                if a!=0:
                    raise Exception("\n\n Wrong, the mode=set but the value had be setted, please use add or reset")
                self._ham[index]+=useham
            elif np.any(T2):
                ind_R*=-1
                useham=useham.transpose().conjugate()
                index=np.argwhere(T1)[[0]]
                a=np.sum(self._ham[index]*useham)
                if a!=0:
                    raise Exception("\n\n Wrong, the mode=set but the value had be setted, please use add or reset")
                self._ham[index]+=useham
            else:
               self._hamR=np.append(self._hamR,[ind_R],axis=0)
               self._ham=np.append(self._ham,[useham],axis=0)
        elif mode.lower()=='reset':
            if np.any(T1):
                index=np.argwhere(T1)[[0]]
                self._ham[index]=self._ham[index]+useham-self._ham[index]*np.array(useham!=0,dtype=complex)
            elif np.any(T2):
                ind_R*=-1
                index=np.argwhere(T1)[[0]]
                useham=useham.transpose().conjugate()
                self._ham[index]=self._ham[index]+useham-self._ham[index]*np.array(useham!=0,dtype=complex)
            else:
               self._hamR=np.append(self._hamR,[ind_R],axis=0)
               self._ham=np.append(self._ham,[useham],axis=0)
        else:
            raise Exception("mode must be set, add or reset, not other")

    def set_onsite(self,onsite,ind=None,mode='set'):
        if ind==None:
            if (len(onsite)!=self._norb):
                raise Exception("\n\nWrong number of site energies")
        if ind!=None:
            if ind<0 or ind>self._norb:
                raise Exception("\n\nIndex ind out of scope.")
        if ind==None:
            if self._dim_r!=0:
                ind_R=np.zeros(self._dim_r,dtype=int)
                for i in range(len(onsite)):
                    self.set_hop(onsite[i],[i,i],ind_R,mode=mode)
            else:
                for i in range(len(onsite)):
                    self.set_hop(onsite[i],[i,i],mode=mode)
                
        else:
            if self._dim_r!=0:
                ind_R=np.zeros(self._dim_r,dtype=int)
                self.set_hop(onsite[ind],[ind,ind],ind_R,mode=mode)
            else:
                self.set_hop(onsite[ind],[ind,ind],mode=mode)

    def set_rmatrix(tmp,ind,R):
        ind=np.array(ind,dtype=int)
        ind_R=np.array(ind_R,dtype=int)
        tmp=np.array(tmp,dtype=complex)
        if tmp.shape!=(self._dim_r,):
            raise Exception("Wrong, the tmp's shape must be (1,dim_r)")
        if len(ind)!=2:
            raise Exception("\n\n length of ind need to be 2")
        if len(ind_R)!=self._dim_r:
            raise Exception("\n\n Wrong, the length of ind_R must equal to dim_r")       


    def k_path(self,kpts,nk,knode_index=False):
        if self._dim_k==0:
            raise Exception("the model's k dimension is zero, do not use k")
        elif self._dim_k==1:
            if kpts=='full':
                k_list=np.array([[0.],[0.5],[1.]])
            elif kpts=='fullc':
                k_list=np.array([[-0.5],[0.],[0.5]])
            elif kpts=='half':
                k_list=np.array([[0.],[0.5]])
            else:
                k_list=np.array(kpts)
        else:
            k_list=np.array(kpts)
        if k_list.shape[1]!=self._dim_k:
            raise Exception("\n\n k-space dimension do not match")
        if nk<k_list.shape[0]:
            raise Exception("\n\n please set more n_k, at least more than the number of k_list")
        n_nodes=k_list.shape[0]
        lat_per=np.copy(self._lat)[self._per]
        k_metric=np.linalg.inv(np.dot(lat_per,lat_per.T))
        k_node=np.zeros(n_nodes,dtype=float)
        for n in range(1,n_nodes):
            dk=k_list[n]-k_list[n-1]
            dklen=np.sqrt(np.dot(dk,np.dot(k_metric,dk)))
            k_node[n]=k_node[n-1]+dklen
        node_index=[0]
        for n in range(1,n_nodes-1):
            frac=k_node[n]/k_node[-1]
            node_index.append(int(round(frac*(nk-1))))
        node_index.append(nk-1)
        k_dist=np.zeros(nk,dtype=float)
        k_vec=np.zeros((nk,self._dim_k),dtype=float)
        k_vec[0]=k_list[0]
        for n in range(1,n_nodes):
            n_i=node_index[n-1]
            n_f=node_index[n]
            kd_i=k_node[n-1]
            kd_f=k_node[n]
            k_i=k_list[n-1]
            k_f=k_list[n]
            for j in range(n_i,n_f+1):
                frac=float(j-n_i)/float(n_f-n_i)
                k_dist[j]=kd_i+frac*(kd_f-kd_i)
                k_vec[j]=k_i+frac*(k_f-k_i)
        if knode_index==False:
            return (k_vec,k_dist,k_node)
        else:
            node_index=np.array(node_index,dtype=int)
            return(k_vec,k_dist,k_node,node_index)
    def gen_ham(self,k_point=None):
        if k_point is None:
            if self._dim_k==0:
                return self._ham[0]
            else:
                raise Exception("\n\n Wrong,the dim_k isn't 0, please input k_point")
        else:
            k_point=np.array(k_point)
            if len(k_point.shape)==2:
                lists=True
                if k_point.shape[1]!=self._dim_k:
                    raise Exception("Wrong, the shape of k_point must equal to dim_k")
            else:
                lists=False
                if k_point.shape[0]!=self._dim_k:
                    raise Exception("Wrong, the shape of k_point must equal to dim_k")
        if np.any(self._hamR[0]!=0):
            raise Exception("Wrong, the first hamR should be zero, please check hamR, or rebuld the model")
        orb=np.array(self._orb)[:,self._per]
        ind_R=self._hamR[1:,self._per]
        useham=self._ham[1:]
        if lists:
            func=functools.partial(gen_ham_try,self._ham[0],useham,ind_R,orb,self._nspin)
            pool=Pool()
            ham=pool.map(func,k_point)
            pool.close()
            pool.join()
            ham=np.array(ham)
        else:
            ham=gen_ham_try(self._ham[0],useham,ind_R,orb,self._nspin,k_point)
        return ham


    def solve_one(self,k_point=None,eig_vectors=False):
        if self._dim_k==0 and k_point!=None:
            raise Exception("Wrong, the dimension of k is zero,please don't set k_point")
        elif k_point is None and self._dim_k!=0:
            raise Exception("Wrong, the dimension of k is not zero, please set k_point")
        if k_point is not None:
            k_point=np.array(k_point)
            if len(k_point.shape)!=1:
                raise Exception("Wrong, the shape of k_point must be (dim_k,), or please use solve_all")
            if len(k_point)!=self._dim_k:
                raise Exception("Wrong, the shape of k_point must equal to dim_k")
            useham=self.gen_ham(k_point)
        else:
            useham=self.gen_ham()
        if eig_vectors:
            (evals,evec)=np.linalg.eigh(useham)
            index=np.argsort(evals)
            evals=evals[index]
            evec=evec[:,index].T
            return (evals,evec)
        else:
            evals=np.linalg.eigvalsh(useham)
            index=np.argsort(evals)
            evals=evals[index]
            return evals


    def solve_all(self,k_point,eig_vectors=False):
        if self._dim_k==0:
            raise Exception("please use solve_one")
        k_point=np.array(k_point)
        if len(k_point.shape)!=2:
            raise Exception("Wrong, the shape of k_point must be (dim_k,), or please use solve_one")
        if k_point.shape[1]!=self._dim_k:
            raise Exception("Wrong, the shape of k_point must equal to dim_k")
        useham=self.gen_ham(k_point)
        if eig_vectors:
            (evals,evec)=np.linalg.eigh(useham)
            evec=evec.transpose((0,2,1))
            return (evals,evec)
        else:
            evals=np.linalg.eigvalsh(useham)
            return evals

    def solve_all_parallel(self,k_point,eig_vectors=False):
        if self._dim_k==0:
            raise Exception("please use solve_one")
        k_point=np.array(k_point)
        if len(k_point.shape)!=2:
            raise Exception("Wrong, the shape of k_point must be (dim_k,), or please use solve_one")
        if k_point.shape[1]!=self._dim_k:
            raise Exception("Wrong, the shape of k_point must equal to dim_k")
        genham=functools.partial(self.gen_ham)
        pool=Pool()
        useham=pool.map(self.gen_ham,k_point)
        pool.close()
        pool.join()
        nk=k_point.shape[0]
        if eig_vectors:
            pool=Pool()
            results=pool.map(np.linalg.eigh,useham)
            pool.close()
            pool.join()
            evals=np.zeros((nk,self._nsta),dtype=float)
            evec=np.zeros((nk,self._nsta,self._nsta),dtype=complex)
            for i in range(nk):
                evals[i]=results[i][0]
                evec[i]=results[i][1]
            evec=evec.transpose((0,2,1))
            return (evals,evec)
        else:
            pool=Pool()
            evals=pool.map(np.linalg.eigvalsh,useham)
            pool.close()
            pool.join()
            evals=np.array(evals,dtype=float)
            return evals
        return evals



    def cut_piece(self,num,fin_dir):
        if self._dim_k==0:
            raise Exception("\n\nModel is already finite")
        if type(num).__name__!='int':
            raise Exception("\n\nArgument num not an integer")
        if num<1:
            raise Exception("\n\nArgument num must be positive!")
        #if num==1 and glue_edge==True:
        #    raise Exception("\n\nCan't have num==1 and glueing of the deges!")
        fin_orb=[]
        fin_atom_position=[]
        fin_atom=[]
        for i in range(num):
            for j in range(self._norb):
                orb_tmp=np.copy(self._orb[j,:])
                orb_tmp[fin_dir]+=float(i)
                fin_orb.append(orb_tmp)
            for j in range(self._natom):
                atom_tmp=np.copy(self._atom_position[j])
                atom_tmp[fin_dir]+=float(i)
                fin_atom_position.append(atom_tmp)
                fin_atom.append(self._atom[j])
        fin_orb=np.array(fin_orb)
        fin_orb[:,fin_dir]*=1/float(num)
        fin_atom_position=np.array(fin_atom_position)
        fin_atom_position[:,fin_dir]*=1/float(num)
        fin_lat=copy.deepcopy(self._lat)
        fin_lat[fin_dir]*=num
        fin_per=copy.deepcopy(self._per)
        if fin_per.count(fin_dir)!=1:
            raise Exception("\n\n Can't make model finite along this direction!")
        fin_per.remove(fin_dir)
        fin_model=tb_model(self._dim_k-1,self._dim_r,fin_lat,fin_orb,fin_per,fin_atom,fin_atom_position,self._nspin)
        nsta=self._nsta
        norb=self._norb
        fin_norb=norb*num
        hamR=np.copy(self._hamR)
        ham=np.copy(self._ham)
        for n in range(num):
            for i0,ind_R in enumerate(hamR):
                if ind_R[fin_dir]<0:
                    ind_R*=-1
                    ham[i0]=ham[i0].conjugate().T
                ind=ind_R[fin_dir]+n
                if ind>=num:
                    continue
                useham=np.zeros((fin_model._nsta,fin_model._nsta),dtype=complex)
                if self._nspin==1:
                    useham[n*nsta:(n+1)*nsta,ind*nsta:(ind+1)*nsta]=ham[i0]
                    if np.all(ind_R[fin_per]==0) and ind!=0:
                        useham[ind*nsta:(ind+1)*nsta,n*nsta:(n+1)*nsta]=ham[i0].conjugate().T
                    if np.any(np.all(fin_model._hamR[:,fin_per]==ind_R[fin_per],axis=1)):
                        index=np.argwhere(np.all(fin_model._hamR[:,fin_per]==ind_R[fin_per],axis=1))[[0]]
                        fin_model._ham[index]+=useham
                    elif np.any(np.all(fin_model._hamR[:,fin_per]==-ind_R[fin_per],axis=1)):
                        index=np.argwhere(np.all(fin_model._hamR[:,fin_per]==-ind_R[fin_per],axis=1))[[0]]
                        useham=useham.transpose().conjugate()
                        fin_model._ham[index]+=useham
                    else:
                        fin_model._ham=np.append(fin_model._ham,[useham],axis=0)
                        fin_model._hamR=np.append(fin_model._hamR,[ind_R],axis=0)
                elif self._nspin==2:
                    useham[n*norb:(n+1)*norb,ind*norb:(ind+1)*norb]=ham[i0,:norb,:norb]
                    useham[n*norb:(n+1)*norb,fin_norb+ind*norb:fin_norb+(ind+1)*norb]=ham[i0,norb:,:norb]
                    useham[fin_norb+n*norb:fin_norb+(n+1)*norb,ind*norb:(ind+1)*norb]=ham[i0,:norb,norb:]
                    useham[fin_norb+n*norb:fin_norb+(n+1)*norb,fin_norb+ind*norb:fin_norb+(ind+1)*norb]=ham[i0,norb:,norb:]
                    if np.all(ind_R[fin_per]==0) and ind!=0:
                        useham[ind*norb:(ind+1)*norb,n*norb:(n+1)*norb]=ham[i0,:norb,:norb].conjugate().T
                        useham[fin_norb+ind*norb:fin_norb+(ind+1)*norb,n*norb:(n+1)*norb]=ham[i0,norb:,:norb].conjugate().T
                        useham[ind*norb:(ind+1)*norb,fin_norb+n*norb:fin_norb+(n+1)*norb]=ham[i0,:norb,norb:].conjugate().T
                        useham[fin_norb+ind*norb:fin_norb+(ind+1)*norb,fin_norb+n*norb:fin_norb+(n+1)*norb]=ham[i0,norb:,norb:].conjugate().T
                    if np.any(np.all(fin_model._hamR[:,fin_per]==ind_R[fin_per],axis=1)):
                        index=np.argwhere(np.all(fin_model._hamR[:,fin_per]==ind_R[fin_per],axis=1))[[0]]
                        fin_model._ham[index]+=useham
                    elif np.any(np.all(fin_model._hamR[:,fin_per]==-ind_R[fin_per],axis=1)):
                        index=np.argwhere(np.all(fin_model._hamR[:,fin_per]==-ind_R[fin_per],axis=1))[[0]]
                        useham=useham.transpose().conjugate()
                        fin_model._ham[index]+=useham
                    else:
                        fin_model._ham=np.append(fin_model._ham,[useham],axis=0)
                        fin_model._hamR=np.append(fin_model._hamR,[ind_R],axis=0)
        return fin_model

    def cut_dot(self,num,num_shape,inf_dir=None):
        if self._dim_k<2 or self._dim_k>3:
            raise Exception("\n\n this function only for 2, 3-dimention system")
        if inf_dir==None and self._dim_k==3:
            raise Exception("\n\n Wrong, when model is 3 dimension, you must give a infinite direction")
        if self._dim_k==2:
            if num_shape==4:
                one_model=self.cut_piece(num+1,self._per[0])
                dot_model=one_model.cut_piece(num+1,self._per[1])
                orb=dot_model._orb*(num+1)/float(num)
                use_orb=((orb[:,self._per[0]])<=1+10**-5)*((orb[:,self._per[1]])<=1+10**-5)
                dot_model._orb=orb[use_orb,:]
                if self._nspin==2:
                    use_orb=np.append(use_orb,use_orb)
                dot_model._ham=dot_model._ham[:,use_orb][:,:,use_orb]
                dot_model._norb=np.sum(use_orb)
                dot_model._nsta=dot_model._norb*dot_model._nspin
                dot_model._lat*=float(num)/float(num+1)
            elif num_shape==3:
                one_model=self.cut_piece(num+1,self._per[0])
                dot_model=one_model.cut_piece(num+1,self._per[1])
                orb=dot_model._orb*(num+1)/float(num)
                use_orb=(orb[:,self._per[0]]+orb[:,self._per[1]])<=1+10**-5
                orb=dot_model._orb[use_orb,:]
                dot_model._orb=orb
                if self._nspin==2:
                    use_orb=np.append(use_orb,use_orb)
                dot_model._ham=dot_model._ham[:,use_orb][:,:,use_orb]
                dot_model._norb=np.sum(use_orb)
                dot_model._nsta=dot_model._norb*dot_model._nspin
                dot_model._lat*=float(num)/float(num+1)
            elif num_shape==6:
                one_model=self.cut_piece(2*num+1,self._per[0])
                dot_model=one_model.cut_piece(2*num+1,one_model._per[0])
                orb=dot_model._orb*(2*num+1)/float(num)
                use_orb=((-orb[:,self._per[0]]+orb[:,self._per[1]])>=-1.-10**-5) \
                    *((-orb[:,self._per[0]]+orb[:,self._per[1]])<=1.+10**-5) \
                    *((orb[:,self._per[0]])<=2.+10**-5)*((orb[:,self._per[1]])<=2.+10**-5)
                orb=dot_model._orb[use_orb,:]
                if self._nspin==2:
                    use_orb=np.append(use_orb,use_orb)
                ham=dot_model._ham[:,use_orb][:,:,use_orb]
                dot_model._orb=orb
                dot_model._ham=ham
                dot_model._norb=np.sum(use_orb)
                dot_model._nsta=dot_model._norb*dot_model._nspin
                dot_model._lat*=float(num)/float(2*num+1)
            return dot_model
        elif self._dim_k==3:
            fin_dir=list(range(3))
            fin_dir.remove(inf_dir)
            if num_shape==4:
                one_model=self.cut_piece(num+1,fin_dir[0])
                dot_model=one_model.cut_piece(num+1,fin_dir[1])
                orb=dot_model._orb*(num+1)/float(num)
                use_orb=((orb[:,fin_dir[0]])<=1+10**-5)*((orb[:,fin_dir[1]])<=1+10**-5)
                dot_model._orb=orb[use_orb,:]
                if self._nspin==2:
                    use_orb=np.append(use_orb,use_orb)
                dot_model._ham=dot_model._ham[:,use_orb][:,:,use_orb]
                dot_model._norb=np.sum(use_orb)
                dot_model._nsta=dot_model._norb*dot_model._nspin
                dot_model._lat*=float(num)/float(num+1)
            elif num_shape==3:
                one_model=self.cut_piece(num+1,fin_dir[0])
                dot_model=one_model.cut_piece(num+1,fin_dir[1])
                orb=dot_model._orb*(num+1)/float(num)
                use_orb=(orb[:,self._per[0]]+orb[:,self._per[1]])<=1+10**-5
                orb=dot_model._orb[use_orb,:]
                dot_model._orb=orb
                if self._nspin==2:
                    use_orb=np.append(use_orb,use_orb)
                dot_model._ham=dot_model._ham[:,use_orb][:,:,use_orb]
                dot_model._norb=np.sum(use_orb)
                dot_model._nsta=dot_model._norb*dot_model._nspin
                dot_model._lat*=float(num)/float(num+1)
            elif num_shape==6:
                one_model=self.cut_piece(2*num+1,delf._per[0])
                dot_model=one_model.cut_piece(2*num+1,one_model._per[0])
                orb=dot_model._orb*(2*num+1)/float(num)
                use_orb=((-orb[:,self._per[0]]+orb[:,self._per[1]])>=-1.-10**-5) \
                    *((-orb[:,self._per[0]]+orb[:,self._per[1]])<=1.+10**-5) \
                    *((orb[:,self._per[0]])<=2.+10**-5)*((orb[:,self._per[1]])<=2.+10**-5)
                orb=dot_model._orb[use_orb,:]
                if self._nspin==2:
                    use_orb=np.append(use_orb,use_orb)
                ham=dot_model._ham[:,use_orb][:,:,use_orb]
                dot_model._orb=orb
                dot_model._ham=ham
                dot_model._norb=np.sum(use_orb)
                dot_model._nsta=dot_model._norb*dot_model._nspin
                dot_model._lat*=float(num)/float(2*num+1)
            return dot_model  
        
    def make_supercell_new(self,U):
        r"new_lat=lat*U"
        if self._dim_r==0:
            raise Exception("\n\n Must have at least one periodic direction to make a super-cell")
        U=np.array(U,dtype=int)
        if U.shape!=(self._dim_r,self._dim_r):
            raise Exception("\n\n Dimension of U must be dim_r*dim_r")
        fin_dir=list(range(self._dim_r))
        for i in self._per:
            fin_dir.remove(i)
        for i in fin_dir:
            if U[i,i]!=1:
                raise Exception("\n\n Diagonal elements of U for non-periodic directions must equal to 1.")
            for j in self._per:
                if U[i,j]!=0 or U[j,i]!=0:
                    raise Exception("\n\n Off-diagonal elements of sc_red_lat for non-periodic directions must equal 0")
        #这里 U 是旋转和变换矩阵, L'=UL.
        det_U=np.linalg.det(U)
        U_inv=np.linalg.inv(U)
        if np.abs(det_U)<1.0E-6:
            raise Exception("\n\n Super-cell latice vectors length/area/volume too close to zero, or zero")
        if det_U*np.linalg.det(self._lat)<0.0:
            raise Exception("\n\n Super-cell lattice vectors need to form right handed system.")
        new_lat=np.dot(U,self._lat)
        max_R=int(np.max(np.abs(U))*self._dim_r)
        min_range=-np.ones(self._dim_r,dtype=int)*max_R
        max_range=np.ones(self._dim_r,dtype=int)*max_R
        (new_orb,new_orb_list,new_atom_position,new_atom)=gen_new_structure(min_range,max_range,self._dim_r,self._atom,self._atom_position,self._orb,U_inv)
        new_orb=shift_to_zero(np.array(new_orb)) #shift_to_zero 是为了防止 1.0 和 0.999999999998 这种情况存在.
        new_orb_list=np.array(new_orb_list,dtype=int)
        new_atom=np.array(new_atom,dtype=int)
        new_atom_position=shift_to_zero(np.array(new_atom_position))
        inter_orb=np.all(new_orb<1,axis=1)*np.all(new_orb>=0,axis=1) #在新的 L' 表示下 0<他<
        inter_atom=np.all(new_atom_position<1,axis=1)*np.all(new_atom_position>=0,axis=1)
        new_orb=new_orb[inter_orb]
        new_orb_list=new_orb_list[inter_orb]
        new_atom=new_atom[inter_atom]
        new_atom_position=new_atom_position[inter_atom]
        new_norb=int(len(new_orb))
        new_natom=int(len(new_atom))
        new_model=tb_model(self._dim_k,self._dim_r,new_lat,new_orb,per=self._per,nspin=self._nspin,atom_list=new_atom,atom_position=new_atom_position) #产生一个空的我们想要的晶胞 model

        ###############################hopping range###############################################
        max_R=np.array(np.ceil(np.max(np.dot(self._hamR,U_inv),axis=0)),dtype=int) #搜索的最大的 R 和最小的 R
        min_R=np.array(np.floor(np.min(np.dot(self._hamR,U_inv),axis=0)),dtype=int)
        new_hamR=gen_hamR_new(min_R,max_R,self._dim_k,self._dim_r,self._per)
        ##########################################################################
        new_model=gen_supercell_hop(new_model,new_hamR,new_orb,new_orb_list,self._ham,self._hamR,self._orb,U)
        return new_model

    def make_supercell(self,U,to_home=True):
        r"new_lat=lat*U 这个最好是在只沿着对角方向的时候再用"
        if self._dim_r==0:
            raise Exception("\n\n Must have at least one periodic direction to make a super-cell")
        U=np.array(U,dtype=int)
        if U.shape!=(self._dim_r,self._dim_r):
            raise Exception("\n\n Dimension of U must be dim_r*dim_r")
        fin_dir=list(range(self._dim_r))
        for i in self._per:
            fin_dir.remove(i)
        for i in fin_dir:
            if U[i,i]!=1:
                raise Exception("\n\n Diagonal elements of U for non-periodic directions must equal to 1.")
            for j in self._per:
                if U[i,j]!=0 or U[j,i]!=0:
                    raise Exception("\n\n Off-diagonal elements of sc_red_lat for non-periodic directions must equal 0")
        det_U=np.linalg.det(U)
        if np.abs(det_U)<1.0E-6:
            raise Exception("\n\n Super-cell latice vectors length/area/volume too close to zero, or zero")
        if det_U*np.linalg.det(self._lat)<0.0:
            raise Exception("\n\n Super-cell lattice vectors need to form right handed system.")
        max_R=int(np.max(np.abs(U))*self._dim_r)
        sc_cands=get_list(self._dim_r,-max_R,max_R+1)
        eps_shift=np.sqrt(2.0)*1.0E-8
        def to_red_sc(red_vec_orig):
            return np.linalg.solve(np.array(U.T,dtype=float), np.array(red_vec_orig,dtype=float))
        sc_vec=[]
        for vec in sc_cands:
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red=to_red_sc(vec).tolist()
            # check if in the interior
            inside=True
            for t in tmp_red:
                if t<=-1.0*eps_shift or t>1.0-eps_shift:
                    inside=False
            if inside==True:
                sc_vec.append(np.array(vec))
        #new_sc=new_sc[index]
        new_sc=np.array(sc_vec)
        num_sc=new_sc.shape[0]
        sc_orb=[]
        sc_atom_position=[]
        sc_atom=[]
        for cur_sc_vec in new_sc:
            for orb in self._orb:
                #sc_orb.append(np.linalg.solve(np.array(U.T,dtype=float),np.array(orb+cur_sc_vec,dtype=float).T).T)
                sc_orb.append(to_red_sc(orb+cur_sc_vec))
            for i,atom in enumerate(self._atom_position):
                sc_atom_position.append(to_red_sc(atom+cur_sc_vec))
                sc_atom.append(self._atom[i])
        sc_atom_position=np.array(sc_atom_position,dtype=float)
        sc_atom=np.array(sc_atom,dtype=int)
        sc_atom=np.array(sc_atom,dtype=int)
        sc_lat=np.dot(U,self._lat)
        sc_tb=tb_model(self._dim_k,self._dim_r,sc_lat,sc_orb,per=self._per,nspin=self._nspin,atom_list=sc_atom,atom_position=sc_atom_position)
        n_sta=self._nsta
        n_orb=self._norb
        for c,cur_sc_vec in enumerate(new_sc):
            for h in range(len(self._hamR)):
                ind_R=copy.deepcopy(self._hamR[h])
                sc_part=np.linalg.solve(np.array(U.T,dtype=float),np.array(ind_R+cur_sc_vec,dtype=float).T).T
                sc_part=np.array(np.floor(sc_part),dtype=int)
                orig_part=ind_R+cur_sc_vec-np.dot(sc_part,U)
                pair_ind=None
                for p,pair_sc_vec in enumerate(new_sc):
                    if np.all(pair_sc_vec==orig_part):
                        if pair_ind!=None:
                            raise Exception("\n\nFound duplicate supercell vector")
                        pair_ind=p
                if pair_ind==None:
                    raise Exception("\n\nDid not find supercell vector")
                ham=self._ham[h]
                useR=sc_part
                useham=np.zeros((sc_tb._nsta,sc_tb._nsta),dtype=complex)
                if self._nspin==1:
                    useham[c*n_sta:(c+1)*n_sta,pair_ind*n_sta:(pair_ind+1)*n_sta]=ham
                    if np.all(sc_part==0) and c!=pair_ind:
                        useham[pair_ind*n_sta:(pair_ind+1)*n_sta,c*n_sta:(c+1)*n_sta]=ham.conjugate().T
                else:
                    norb=sc_tb._norb
                    useham[c*n_orb:(c+1)*n_orb,pair_ind*n_orb:(pair_ind+1)*n_orb]=ham[:n_orb,:n_orb]
                    useham[norb+c*n_orb:norb+(c+1)*n_orb,norb+pair_ind*n_orb:norb+(pair_ind+1)*n_orb]=ham[n_orb:,n_orb:]
                    useham[c*n_orb:(c+1)*n_orb,norb+pair_ind*n_orb:norb+(pair_ind+1)*n_orb]=ham[:n_orb,n_orb:]
                    useham[norb+c*n_orb:norb+(c+1)*n_orb,pair_ind*n_orb:(pair_ind+1)*n_orb]=ham[n_orb:,:n_orb]
                    if np.all(sc_part==0) and c!=pair_ind:
                        useham[pair_ind*n_orb:(pair_ind+1)*n_orb,c*n_orb:(c+1)*n_orb]=ham[:n_orb,:n_orb].conjugate().T
                        useham[norb+pair_ind*n_orb:norb+(pair_ind+1)*n_orb,norb+c*n_orb:norb+(c+1)*n_orb]=ham[n_orb:,n_orb:].conjugate().T
                if np.any(np.all(sc_tb._hamR==sc_part,axis=1)):
                    index=np.argwhere(np.all(sc_tb._hamR==sc_part,axis=1))[[0]]
                    sc_tb._ham[index]+=useham
                elif np.any(np.all(sc_tb._hamR==-sc_part,axis=1)):
                    index=np.argwhere(np.all(sc_tb._hamR==-sc_part,axis=1))[[0]]
                    sc_tb._ham[index]+=useham.conjugate().T
                else:
                    sc_tb._ham=np.append(sc_tb._ham,[useham],axis=0)
                    sc_tb._hamR=np.append(sc_tb._hamR,[sc_part],axis=0)
        if to_home:
            sc_tb.shift_to_home()
        return sc_tb

    def remove_dim(self,remove_k,value_k=0):
        if self._dim_k==0:
            raise Exception("\n\n Can not reduce dimensionality even further!")
        self._per.remove(remove_k)
        dim_k=len(self._per)
        if dim_k!=self._dim_k-1:
            raise Exception("\n\n Specified wrong dimension to reduce!")
        self._dim_k=dim_k
        rv=self._orb[:,remove_k]
        if value_k!=0:
            U=np.diag(np.exp(2.j*np.pi*rv*value_k))
            if self._nspin==2:
                U=np.kron([[1,0],[0,1]],U)
            for i in range(self._ham.shape[0]):
                self._ham[i]=np.dot(self._ham[i],U)
                self._ham[i]=np.dot(U.T.conjugate(),self._ham[i])
                self._hamR[i,remove_k]=0


    def shift_to_home(self):
        #将原胞外的原子移动到原胞内
        new_ham=np.copy(self._ham)
        new_hamR=np.copy(self._hamR)
        self._atom_position=shift_to_zero(self._atom_position)%1.0
        for i in range(self._norb):
            cur_orb=self._orb[i]
            round_orb=shift_to_zero(np.array(cur_orb))%1.0
            dis_vec=np.array(np.round(cur_orb-round_orb),dtype=int)
            if np.any(dis_vec!=0):
                self._orb[i]-=np.array(dis_vec,dtype=float)
                for i0 in (self._hamR-dis_vec):
                    if np.any(np.all(self._hamR==i0,axis=1)):
                        index_i=np.argwhere(np.all(self._hamR==i0,axis=1))
                        if np.any(np.all(new_hamR==i0,axis=1)):
                            index=np.all(new_hamR==i0,axis=1)
                            new_ham[index,i,:]=self._ham[index_i,i,:]
                            if self._nspin==2:
                                new_ham[index,i+self._norb,:]=self._ham[index_i,i+self._norb,:]
                        elif np.any(np.all(new_hamR==-i0,axis=1)):
                            index=np.all(new_hamR==-i0,axis=1)
                            new_ham[index,:,i]=self._ham[index_i,i,:].T.conjugate()
                            if self._nspin==2:
                                new_ham[index,i+self._norb,:]=self._ham[index_i,:,i+self._norb].T.conjugate()
                        else:
                            new_hamR=np.append(new_hamR,[i0],axis=0)
                            ham0=np.zeros((self._nsta,self._nsta),dtype=complex)
                            ham0[i,:]=self._ham[index_i,i,:]
                            if self._nspin==2:
                                ham0[i+self._norb,:]=self._ham[index_i,i+self._norb,:]
                            new_ham=np.append(new_ham,[ham0],axis=0)
                    elif np.any(np.all(self._hamR==-i0,axis=1)): 
                        index_i=np.argwhere(np.all(self._hamR==-i0,axis=1))
                        if np.any(np.all(new_hamR==i0,axis=1)):
                            index=np.all(new_hamR==i0,axis=1)
                            new_ham[index,i,:]=self._ham[index_i,:,i].T.conjugate()
                            if self._nspin==2:
                                new_ham[index,i+self._norb,:]=self._ham[index_i,:,i+self._norb].T.conjugate
                        elif np.any(np.all(new_hamR==-i0,axis=1)):
                            index=np.all(new_hamR==-i0,axis=1)
                            new_ham[index,:,i]=self._ham[index_i,:,i]
                            if self._nspin==2:
                                new_ham[index,:,i+self._norb]=self._ham[index_i,:,i+self._norb]
                        else:
                            new_hamR=np.append(new_hamR,[-i0],axis=0)
                            ham0=np.zeros((self._nsta,self._nsta),dtype=complex)
                            ham0[i,:]=self._ham[index_i,:,i].T.conjugate()
                            if self._nspin==2:
                                ham0[i+self._norb,:]=self._ham[index_i,:,i+self._norb].conjugate().T
                            new_ham=np.append(new_ham,[ham0],axis=0)

                for j0 in (self._hamR+dis_vec):
                    if np.any(np.all(self._hamR==j0,axis=1)):
                        index_j=np.argwhere(np.all(self._hamR==j0,axis=1))
                        if np.any(np.all(new_hamR==j0,axis=1)):
                            index=np.all(new_hamR==j0,axis=1)
                            new_ham[index,:,i]=self._ham[index_j,:,i]
                            if self._nspin==2:
                                new_ham[index,:,i+self._norb]=self._ham[index_j,:,i+self._norb]
                        elif np.any(np.all(new_hamR==-j0,axis=1)):
                            index=np.all(new_hamR==-j0,axis=1)
                            new_ham[index,i,:]=self._ham[index_j,:,i].conjugate().T
                            if self._nspin==2:
                                new_ham[index,i+self._norb,:]=self._ham[index_j,:,i+self._norb].conjugate().T
                        else:
                            new_hamR=np.append(new_hamR,[j0],axis=0)
                            ham0=np.zeros((self._nsta,self._nsta),dtype=complex)
                            ham0[i,:]=self._ham[index_j,:,i]
                            if self._nspin==2:
                                ham0[i+self._norb,:]=self._ham[index_j,:,i+self._norb]
                            new_ham=np.append(new_ham,[ham0],axis=0)
                    elif np.any(np.all(self._hamR==-j0,axis=1)):
                        index_j=np.argwhere(np.all(self._hamR==-j0,axis=1))
                        if np.any(np.all(new_hamR==j0,axis=1)):
                            index=np.all(new_hamR==j0,axis=1)
                            new_ham[index,:,i]=self._ham[index_j,i,:].conjugate().T
                            if self._nspin==2:
                                new_ham[index,:,i+self._norb]=self._ham[index_j,i+self._norb,:].conjugate().T
                        elif np.any(np.all(new_hamR==-j0,axis=1)):
                            index=np.all(new_hamR==-j0,axis=1)
                            new_ham[index,i,:]=self._ham[index_j,:,i]
                            if self._nspin==2:
                                new_ham[index,i+self._norb,:]=self._ham[index_j,:,i+self._norb]
                        else:
                            new_hamR=np.append(new_hamR,[-j0],axis=0)
                            ham0=np.zeros((self._nsta,self._nsta),dtype=complex)
                            ham0[i,:]=self._ham[index_j,:,i]
                            if self._nspin==2:
                                ham0[i+self._norb,:]=self._ham[index_j,i+self._norb,:].conjugate().T
                            new_ham=np.append(new_ham,[ham0],axis=0)
        self._ham=new_ham  
        
    def visualize(self,dir_first,dir_second=None,draw_hoppings=False,eig_dr=None,ph_color="black"):
        r"""

        Rudimentary function for visualizing tight-binding model geometry,
        hopping between tight-binding orbitals, and electron eigenstates.

        If eigenvector is not drawn, then orbitals in home cell are drawn
        as red circles, and those in neighboring cells are drawn with
        different shade of red. Hopping term directions are drawn with
        green lines connecting two orbitals. Origin of unit cell is
        indicated with blue dot, while real space unit vectors are drawn
        with blue lines.

        If eigenvector is drawn, then electron eigenstate on each orbital
        is drawn with a circle whose size is proportional to wavefunction
        amplitude while its color depends on the phase. There are various
        coloring schemes for the phase factor; see more details under
        *ph_color* parameter. If eigenvector is drawn and coloring scheme
        is "red-blue" or "wheel", all other elements of the picture are
        drawn in gray or black.

        :param dir_first: First index of Cartesian coordinates used for
          plotting.

        :param dir_second: Second index of Cartesian coordinates used for
          plotting. For example if dir_first=0 and dir_second=2, and
          Cartesian coordinates of some orbital is [2.0,4.0,6.0] then it
          will be drawn at coordinate [2.0,6.0]. If dimensionality of real
          space (*dim_r*) is zero or one then dir_second should not be
          specified.

        :param eig_dr: Optional parameter specifying eigenstate to
          plot. If specified, this should be one-dimensional array of
          complex numbers specifying wavefunction at each orbital in
          the tight-binding basis. If not specified, eigenstate is not
          drawn.

        :param ph_color: Optional parameter determining the way
          eigenvector phase factors are translated into color. Default
          value is "black". Convention of the wavefunction phase is as
          in convention 1 in section 3.1 of :download:`notes on
          tight-binding formalism  <misc/pythtb-formalism.pdf>`.  In
          other words, these wavefunction phases are in correspondence
          with cell-periodic functions :math:`u_{n {\bf k}} ({\bf r})`
          not :math:`\Psi_{n {\bf k}} ({\bf r})`.

          * "black" -- phase of eigenvectors are ignored and wavefunction
            is always colored in black.

          * "red-blue" -- zero phase is drawn red, while phases or pi or
            -pi are drawn blue. Phases in between are interpolated between
            red and blue. Some phase information is lost in this coloring
            becase phase of +phi and -phi have same color.

          * "wheel" -- each phase is given unique color. In steps of pi/3
            starting from 0, colors are assigned (in increasing hue) as:
            red, yellow, green, cyan, blue, magenta, red.

        :returns:
          * **fig** -- Figure object from matplotlib.pyplot module
            that can be used to save the figure in PDF, EPS or similar
            format, for example using fig.savefig("name.pdf") command.
          * **ax** -- Axes object from matplotlib.pyplot module that can be
            used to tweak the plot, for example by adding a plot title
            ax.set_title("Title goes here").

        Example usage::

          # Draws x-y projection of tight-binding model
          # tweaks figure and saves it as a PDF.
          (fig, ax) = tb.visualize(0, 1)
          ax.set_title("Title goes here")
          fig.savefig("model.pdf")

        See also these examples: :ref:`edge-example`,
        :ref:`visualize-example`.

        """

        # check the format of eig_dr
        if not (eig_dr is None):
            if eig_dr.shape!=(self._nsta,):
                raise Exception("\n\nWrong format of eig_dr! Must be array of size norb.")

        # check that ph_color is correct
        if ph_color not in ["black","red-blue","wheel"]:
            raise Exception("\n\nWrong value of ph_color parameter!")

        # check if dir_second had to be specified
        if dir_second==None and self._dim_r>1:
            raise Exception("\n\nNeed to specify index of second coordinate for projection!")

        # start a new figure
        import pylab as plt
        fig=plt.figure(figsize=[plt.rcParams["figure.figsize"][0],
                                plt.rcParams["figure.figsize"][0]])
        ax=fig.add_subplot(111, aspect='equal')

        def proj(v):
            "Project vector onto drawing plane"
            coord_x=v[dir_first]
            if dir_second==None:
                coord_y=0.0
            else:
                coord_y=v[dir_second]
            return [coord_x,coord_y]

        def to_cart(red):
            "Convert reduced to Cartesian coordinates"
            return np.dot(red,self._lat)

        # define colors to be used in plotting everything
        # except eigenvectors
        if (eig_dr is None) or ph_color=="black":
            c_cell="b"
            c_orb="r"
            c_nei=[0.85,0.65,0.65]
            c_hop="g"
        else:
            c_cell=[0.4,0.4,0.4]
            c_orb=[0.0,0.0,0.0]
            c_nei=[0.6,0.6,0.6]
            c_hop=[0.0,0.0,0.0]
        # determine color scheme for eigenvectors
        def color_to_phase(ph):
            if ph_color=="black":
                return "k"
            if ph_color=="red-blue":
                ph=np.abs(ph/np.pi)
                return [1.0-ph,0.0,ph]
            if ph_color=="wheel":
                if ph<0.0:
                    ph=ph+2.0*np.pi
                ph=6.0*ph/(2.0*np.pi)
                x_ph=1.0-np.abs(ph%2.0-1.0)
                if ph>=0.0 and ph<1.0: ret_col=[1.0 ,x_ph,0.0 ]
                if ph>=1.0 and ph<2.0: ret_col=[x_ph,1.0 ,0.0 ]
                if ph>=2.0 and ph<3.0: ret_col=[0.0 ,1.0 ,x_ph]
                if ph>=3.0 and ph<4.0: ret_col=[0.0 ,x_ph,1.0 ]
                if ph>=4.0 and ph<5.0: ret_col=[x_ph,0.0 ,1.0 ]
                if ph>=5.0 and ph<=6.0: ret_col=[1.0 ,0.0 ,x_ph]
                return ret_col

        # draw origin
        ax.plot([0.0],[0.0],"o",c=c_cell,mec="w",mew=0.0,zorder=7,ms=4.5)

        # first draw unit cell vectors which are considered to be periodic
        for i in self._per:
            # pick a unit cell vector and project it down to the drawing plane
            vec=proj(self._lat[i])
            ax.plot([0.0,vec[0]],[0.0,vec[1]],"-",c=c_cell,lw=1.5,zorder=7)

        # now draw all orbitals
        for i in range(self._norb):
            # find position of orbital in cartesian coordinates
            pos=to_cart(self._orb[i])
            pos=proj(pos)
            ax.plot([pos[0]],[pos[1]],"o",c=c_orb,mec="w",mew=0.0,zorder=10,ms=4.0)

        orb_list=np.zeros(self._natom,dtype=int)
        orb_list=gen_orb_list(orb_list,self._natom,self._atom)
        draw_hamR=np.zeros((1,self._dim_r),dtype=int)
        draw_orb=np.zeros((1,2),dtype=int)
        if draw_hoppings==True: #画hopping
            for r,R in enumerate(self._hamR):
                for i in range(self._norb):
                    for j in range(self._norb):
                        if np.abs(self._ham[r][i,j])!=0:
                            draw_hamR=np.append(draw_hamR,[R],axis=0) #找到非0 hopping
                            draw_orb=np.append(draw_orb,[[i,j]],axis=0)
            for s0,h in enumerate(draw_hamR):
                for s in range(2):
                    i=draw_orb[s0,0]
                    j=draw_orb[s0,1]
                    pos_i=np.copy(self._orb[i])
                    pos_j=np.copy(self._orb[j])
                    if self._dim_k!=0:
                        if s==0:
                            pos_j[self._per]=pos_j[self._per]+h[self._per]
                        if s==1:
                            pos_i[self._per]=pos_i[self._per]-h[self._per]
                    pos_i=np.array(proj(to_cart(pos_i)))
                    pos_j=np.array(proj(to_cart(pos_j)))
                    all_pnts=np.array([pos_i,pos_j]).T
                    ax.plot(all_pnts[0],all_pnts[1],"-",c=c_hop,lw=0.75,zorder=8)
                    ax.plot([pos_i[0]],[pos_i[1]],"o",c=c_nei,zorder=9,mew=0.0,ms=4.0,mec="w")
                    ax.plot([pos_j[0]],[pos_j[1]],"o",c=c_nei,zorder=9,mew=0.0,ms=4.0,mec="w")
        # draw hopping terms

        # now draw the eigenstate
        if not (eig_dr is None):
            if self._nspin==2:
                orb=np.append(self._orb,self._orb,axis=0)
            else:
                orb=self._orb
            for i in range(self._nsta):
                # find position of orbital in cartesian coordinates

                pos=to_cart(orb[i])
                pos=proj(pos)
                # find norm of eigenfunction at this point
                nrm=(eig_dr[i]*eig_dr[i].conjugate()).real
                # rescale and get size of circle
                nrm_rad=2.0*nrm*float(self._norb)
                # get color based on the phase of the eigenstate
                phase=np.angle(eig_dr[i])
                c_ph=color_to_phase(phase)
                ax.plot([pos[0]],[pos[1]],"o",c=c_ph,mec="w",mew=0.0,ms=nrm_rad,zorder=11,alpha=0.8)

        # center the image
        #  first get the current limit, which is probably tight
        xl=ax.set_xlim()
        yl=ax.set_ylim()
        # now get the center of current limit
        centx=(xl[1]+xl[0])*0.5
        centy=(yl[1]+yl[0])*0.5
        # now get the maximal size (lengthwise or heightwise)
        mx=max([xl[1]-xl[0],yl[1]-yl[0]])
        # set new limits
        extr=0.05 # add some boundary as well
        ax.set_xlim(centx-mx*(0.5+extr),centx+mx*(0.5+extr))
        ax.set_ylim(centy-mx*(0.5+extr),centy+mx*(0.5+extr))

        # return a figure and axes to the user
        return (fig,ax)


    def output(self,path=".",prefix="wannier90"):
        R0=np.append(self._hamR,-self._hamR[1:],axis=0)
        n_R=len(R0)
        if self._dim_r==2:
            R=np.append(R0,np.array([np.zeros(n_R,dtype=int)]).T,axis=1)
        elif self._dim_r==1:
            R=np.append(R0,np.zeros((n_R,2),dtype=int),axis=1)
        else:
            R=R0
        ham=np.append(self._ham,self._ham[1:].transpose(0,2,1).conjugate(),axis=0)
        #arg=np.argsort(np.dot(R,[100,10,1]),axis=0)
        #R=R[arg]
        #ham=ham[arg]
        n_line=int(n_R/15)
        n0=int(n_R%15)
        f=open(path+"/"+prefix+"_hr.dat","w")
        f.write("writen by pythtb\n")
        f.write("         "+str(int(self._nsta))+"\n")
        f.write("         "+str(int(n_R))+"\n")
        for i in range(n_line):
            f.write("    1    1    1    1    1    1    1    1    1    1    1    1    1    1    1\n")
        for i in range(n0):
            f.write("    1")
        f.write("\n")
        for i in range(n_R):
            for j in range(self._nsta):
                for k in range(self._nsta):
                    for i0 in range(3):
                        if R[i,i0]<0:
                            f.write("   "+str(R[i,i0]))
                        else:
                            f.write("    "+str(R[i,i0]))
                    f.write("    "+str(j+1))  
                    f.write("    "+str(k+1))
                    if "-" in "%.6f"%ham[i,j,k].real:
                        f.write("   %.6f"%ham[i,j,k].real)
                    else:
                        f.write("    %.6f"%ham[i,j,k].real)
                    if "-" in "%.6f"%ham[i,j,k].imag:
                        f.write("   %.6f\n"%ham[i,j,k].imag)
                    else:
                        f.write("    %.6f\n"%ham[i,j,k].imag)
        f.close() 
        n0=int(self._norb+self._natom)
        f=open(path+"/"+prefix+"_centre.xyz","w")
        f.write("    "+str(n0)+" \n")
        f.write("writen by pythtb\n")
        if self._dim_r!=3:
            orb=np.append(self._orb,np.zeros((self._norb,3-self._dim_r)),axis=1)
            atom_position=np.append(self._atom_position,np.zeros((self._natom,3-self._dim_r)),axis=1)
        else:
            orb=self._orb
            atom_position=self._atom_position
        for i in range(self._norb):
            f.write("X")
            for j in range(3):
                f.write("         ")
                if "-" in "%.6f"%orb[i,j]:
                    f.write("%.6f"%orb[i,j])
                else:
                    f.write(" %.6f"%orb[i,j])
            f.write("\n")
        if self._nspin==2:
            for i in range(self._norb):
                f.write("X")
                for j in range(3):
                    f.write("         ")
                    if "-" in "%.6f"%orb[i,j]:
                        f.write("%.6f"%orb[i,j])
                    else:
                        f.write(" %.6f"%orb[i,j])
                f.write("\n")
        for i in range(self._natom):
            f.write("H")
            for j in range(3):
                f.write("         ")
                if "-" in "%.6f"%atom_position[i,j]:
                    f.write("%.6f"%atom_position[i,j])
                else:
                    f.write(" %.6f"%atom_position[i,j])
            f.write("\n")
        f.close()

    def solve_dos(self,mesh_arr,energy_range,n_e,method='Gaussian',sigma=0.01):
        start_k=np.zeros(self._dim_k,dtype=float)
        k_points=gen_mesh_arr(mesh_arr,self._dim_k,start_k)
        evals=self.solve_all_parallel(k_points)
        E0=np.linspace(energy_range[0],energy_range[1],n_e)
        if 'G' in method:
            center=evals.ravel()
            dos=np.zeros(n_e,dtype=float)
            dos=Gauss(E0,dos,center,sigma)

        return dos



    def k_path_unfold(self,U,kvec):
        #这个函数是用来将能带路径由原胞的倒空间折叠到超胞的倒空间
        inv_U=np.linalg.inv(U)
        lat=self._lat
        u_lat=np.dot(inv_U,lat)#unfold unit cell latters
        if self._dim_r==3:
            V=np.linalg.det(lat) #超胞体积
            u_V=np.linalg.det(u_lat) #原胞体积
            K=np.array([np.cross(lat[1],lat[2]),np.cross(lat[2],lat[0]),np.cross(lat[0],lat[1])])/V*(np.pi*2) #超胞的K空间基矢
            u_K=np.array([np.cross(u_lat[1],u_lat[2]),np.cross(u_lat[2],u_lat[0]),np.cross(u_lat[0],u_lat[1])])/u_V*(np.pi*2) #原胞的 K 空间基矢
        elif self._dim_r==2:
            V=np.linalg.det(lat)
            u_V=np.linalg.det(u_lat)
            K=np.array([[lat[1,1],-lat[1,0]],[-lat[0,1],lat[0,0]]])/V*np.pi*2 #二维情况下超胞的倒空间基矢
            u_K=np.array([[u_lat[1,1],-u_lat[1,0]],[-u_lat[0,1],u_lat[0,0]]])/u_V*np.pi*2
        kvec0=np.dot(kvec,u_K) #分数坐标乘以原胞基矢,得到倒空间坐标
        kvec0=np.dot(kvec0,np.linalg.inv(K))#接下来乘以超胞的倒空间基矢, 得到超胞的倒空间分数坐标
        kvec0-=np.floor(kvec0) #移动到 [0,1] 区间
        #kvec0-=0.5 #移动到 [-0.5,0.5] 区间
        return kvec0

    def gen_orb_math(self,U,judge):
        #这个函数是用来产生超胞轨道和原胞轨道的对应列表
        inv_U=np.linalg.inv(U)
        U_det=int(np.linalg.det(U))
        lat=np.dot(inv_U,self._lat) #原胞的基矢
        orb=shift_to_zero(np.dot(self._orb,U)[:,self._per]) #原胞下轨道的分数坐标
        orb0=shift_to_zero(orb%1) #将其归一化
        orb_R=np.floor(orb) #将超胞中的轨道确定属于原胞中的哪一个原胞
        orb_arg=np.arange(self._norb,dtype=int) #产生一个空的轨道, 用于安放超胞对应到原胞的顺序.
        atom_arg=np.arange(self._natom,dtype=int)
        ##################define atom list####################
        super_atom_position=np.copy(self._atom_position) #超胞的原子位置
        atom_position=shift_to_zero(np.dot(super_atom_position,U))%1 #对应原胞的原子位置
        index=np.ones(self._natom,dtype=bool) #用来排除 atom_position 中相同的原子.
        for i in range(self._natom-1):
            if index[i]==True:
                for j in range(i+1,self._natom):
                    if np.linalg.norm(atom_position[i]-atom_position[j])<judge and index[j]==True: #判断后面的电脑是否与前面的位置重合, 若重合, 删除排序靠后的. 若 index=False, 说明已经被排除, 不会被再次考虑
                        index[j]=False
        unit_atom_position=atom_position[index]  # 将重复的原子删除
        unit_atom=self._atom[index] #将 orbitital 对应到原子位置上
        unit_natom=len(unit_atom)
        for i in range(unit_natom):
            for j in range(self._natom):
                if np.linalg.norm(unit_atom_position[i]-atom_position[j])<judge:
                    atom_arg[j]=i
        suer_atom_r=np.dot(super_atom_position,self._lat)# 实空间下超胞原子位置
        atom_r=np.dot(atom_position,lat) #实空间下反折叠后超胞的原子位置
        unit_atom_r=np.dot(super_atom_position,self._lat)# 实空间下原胞原子位置
        unit_orb_list=np.zeros(unit_natom,dtype=int) #原胞中, 第 n 个原子的第一个态在 orb 中的位置
        super_orb_list=np.zeros(self._natom,dtype=int) #超胞中, 第 n 个原子的第一个态在 orb 中的位置
        a=0
        for i in range(unit_natom):
            unit_orb_list[i]=a
            for j in range(unit_atom[i]):
                a+=1
        a=0
        for i in range(self._natom):
            super_orb_list[i]=a
            for j in range(self._atom[i]):
                a+=1 
        for i in range(self._natom):
            for j in range(self._atom[i]):
                orb_arg[super_orb_list[i]+j]=unit_orb_list[atom_arg[i]]+j
        norb_unfold=len(orb_arg)
        return (orb_arg,norb_unfold)

    def unfold_main(self,U,k_fold,k_unfold,energy_window,orb_arg,norb_unfold,eta=0.01):
        if self._nspin==2:
            orb_arg*=2
            orb_arg=np.append(orb_arg,orb_arg)
        U_inv=np.linalg.inv(U)
        U=np.array(U,dtype=float)
        orb=np.dot(self._orb,U)[:,self._per]
        lat_unfold=np.dot(U_inv,self._lat)[:,self._per]
        lat_fold=np.array(self._lat)
        nk=len(k_fold)
        n_e=len(energy_window)
        G=np.zeros((n_e,nk,self._nsta),dtype=float)
        B=np.zeros((n_e,nk,norb_unfold),dtype=float)
        (eval,evec)=self.solve_all_parallel(k_fold,eig_vectors=True)
        for i,E in enumerate(energy_window):
            G[i,:]=-1/np.pi*np.imag(1/((E+1.j*eta)-eval))
        nspin=self._nspin
        phase=np.dot(k_fold,self._orb[:,self._per].T)
        r=np.dot(k_unfold,orb.T)-phase
        if nspin==2:
            r=np.append(r,r,axis=1)
        Us=np.exp(2*np.pi*1.j*r)
        for j in range(norb_unfold):
            use_index=(orb_arg==j)
            U0=np.sum(evec[:,:,use_index]*Us[:,None,use_index],axis=2)
            B[:,:,j]=np.sum((np.abs(U0[None,:])**2)*G,axis=2)
        B=np.sum(B[::-1],axis=2)
        B=(B-np.min(B))/(np.max(B)-np.min(B))
        return B

    def unfold(self,U,kvec,energy_window,eta=0.01,judge=0.1):
        self.shift_to_home()
        U=np.array(U,dtype=float)
        k_unfold=kvec
        if self._dim_r>self._dim_k:
            U0=U[self._per][:,self._per]
        #k_fold=self.k_path_unfold(U,kvec)
        k_fold=np.dot(kvec,U.T)
        ###################################################
        (orb_arg,norb_unfold)=self.gen_orb_math(U,judge)
        B=self.unfold_main(U,k_fold,k_unfold,energy_window,orb_arg,norb_unfold,eta)
        return B

    def shift_to_atom(self):
        orb=0
        for i in range(self._natom):
            for j in range(self._atom[i]):
                self._orb[orb]=self._atom_position[i]
                orb+=1
    def gen_v(self,k_point=None):
        if k_point is None:
            if self._dim_k==0:
                return self._ham[0]
            else:
                raise Exception("\n\n Wrong,the dim_k isn't 0, please input k_point")
        else:
            k_point=np.array(k_point)
            if len(k_point.shape)==2:
                lists=True
                if k_point.shape[1]!=self._dim_k:
                    raise Exception("Wrong, the shape of k_point must equal to dim_k")
            else:
                lists=False
                if k_point.shape[0]!=self._dim_k:
                    raise Exception("Wrong, the shape of k_point must equal to dim_k")
        orb=np.array(self._orb)[:,self._per]
        ind_R=self._hamR[1:,self._per]
        useham=self._ham[1:]
        ind_R0=np.dot(self._hamR,self._lat)[1:]
        real_orb=np.dot(self._orb,self._lat)
        U0=np.zeros((self._dim_r,self._norb,self._norb),dtype=complex)
        for i in range(self._norb):
            for j in range(self._norb):
                U0[:,i,j]=-real_orb[i]+real_orb[j]
        if self._nspin==2:
            U0=np.kron([[1,1],[1,1]],U0)
        if lists:
            ham=np.zeros((len(k_point),self._nsta,self._nsta),dtype=np.complex)
            for i,k in enumerate(k_point):
                ham[i]=gen_ham_v(self._ham[0],useham,ind_R,orb,ind_R0,U0,self._nspin,k)
        else:
            ham=gen_ham_v(self._ham[0],useham,ind_R,orb,ind_R0,U0,self._nspin,k_point)
        return ham


    def berry_curvature(self,k_path,direction_1,direction_2,T,og=0,k_B=8.617e-5,spin=0,eta=1e-3,Ef=0):
        direction_1=np.array(direction_1)
        direction_2=np.array(direction_2)
        t0=time.time()
        (band,evec)=self.solve_all_parallel(k_path,eig_vectors=True)
        t1=time.time()
        pool=Pool()
        v=pool.map(self.gen_v,k_path)
        pool.close()
        pool.join()
        t2=time.time()
        if self._nspin==2:
            if spin==0:
                s=np.array([[1,0],[0,1]],dtype=complex)/2
            elif spin==1:
                s=np.array([[0,1],[1,0]],dtype=complex)/2
            elif spin==2:
                s=np.array([[0,-1.j],[1.j,0]],dtype=complex)/2
            elif spin==3:
                s=np.array([[1,0],[0,-1]],dtype=complex)/2
            X=np.kron(s,np.eye(self._norb))
            J=(np.matmul(X,v)+np.matmul(v,X))/2
        else:
            J=v
        J=np.einsum("krij,r->kij",J,direction_1)
        v=np.einsum("krij,r->kij",v,direction_2)
        Omega_n=np.zeros((len(k_path),self._nsta),dtype=float)
        Omega_n=calculate_omegan(J,v,evec,band,self._nsta,Omega_n,og,eta)
        t3=time.time()
        Omega=np.zeros(len(k_path),dtype=float)
        if T>1:
            beta=1/(T*k_B)
            fermi_dirac=1/(np.exp(beta*band-Ef)+1)
            Omega=np.sum(fermi_dirac*Omega_n,axis=1)
        else:
            fermi_dirac=np.array(band<=Ef,dtype=float)
            Omega=np.sum(Omega_n*fermi_dirac,axis=1)
        t4=time.time()
        print("%f,%f,%f,%f"%(t1-t0,t2-t1,t3-t2,t4-t3))
        return Omega

    def berry_curvature_onek(self,k_use,direction_1,direction_2,T,og=0,k_B=8.617e-5,spin=0,eta=1e-3,Ef=0):
        direction_1=np.array(direction_1)
        direction_2=np.array(direction_2)
        (band,evec)=self.solve_one(k_use,eig_vectors=True)
        v=self.gen_v(k_use)
        if self._nspin==2:
            if spin==0:
                s=np.array([[1,0],[0,1]],dtype=complex)/2
            elif spin==1:
                s=np.array([[0,1],[1,0]],dtype=complex)/2
            elif spin==2:
                s=np.array([[0,-1.j],[1.j,0]],dtype=complex)/2
            elif spin==3:
                s=np.array([[1,0],[0,-1]],dtype=complex)/2
            X=np.kron(s,np.eye(self._norb))
            J=(np.matmul(X,v)+np.matmul(v,X))/2
        else:
            J=v
        J=np.einsum("rij,r->ij",J,direction_1)
        v=np.einsum("rij,r->ij",v,direction_2)
        A1=np.matmul(evec.conj(),np.matmul(J,evec.T))
        A2=np.matmul(evec.conj(),np.matmul(v,evec.T))
        U0=np.zeros((self._nsta,self._nsta),dtype=complex)
        for i in range(self._nsta):
            for j in range(self._nsta):
                U0[i,j]=1/((band[i]-band[j])**2-(og+1.j*eta)**2)
        omega_n=-2*np.diag(np.matmul((A1*U0),A2)).imag
        if T>1:
            beta=1/(T*k_B)
            fermi_dirac=1/(np.exp(beta*band-Ef)+1)
            Omega=np.sum(fermi_dirac*omega_n)
        else:
            fermi_dirac=np.array(band<=Ef,dtype=float)
            Omega=np.sum(omega_n*fermi_dirac)
        return Omega

    def spin_hall_conductivity(self,k_mesh,dir_1,dir_2,T,og=0,k_B=0.025852,spin=0,eta=1e-3,Ef=0):
        start_k=np.zeros(self._dim_k,dtype=float)
        if len(k_mesh)!=self._dim_k:
            raise Exception("Wrong, the dimension of k_mesh must equal to dim_k")
        func=functools.partial(self.berry_curvature_onek,direction_1=dir_1,direction_2=dir_2,T=T,og=og,k_B=k_B,spin=spin,eta=eta,Ef=Ef)
        kpt=gen_mesh_arr(k_mesh,self._dim_k,start_k)
        nk=len(kpt)
        pool=Pool()
        Omega=pool.map(func,kpt)
        pool.close()
        pool.join()
        sigma=np.sum(Omega)/nk*(2*np.pi)**self._dim_k/np.linalg.det(self._lat)
        return sigma 

    def berry_curvature_onek_forparallel(self,x,y,z,direction_1,direction_2,T,og=0,k_B=8.617e-5,spin=0,eta=1e-3,Ef=0):
        if self._dim_k==1:
            return self.berry_curvature_onek([x],direction_1,direction_2,T,og=og,k_B=k_B,spin=spin,eta=eta,Ef=Ef)
        elif self._dim_k==2:
            return self.berry_curvature_onek([x,y],direction_1,direction_2,T,og=og,k_B=k_B,spin=spin,eta=eta,Ef=Ef)
        elif self._dim_k==3:
            return self.berry_curvature_onek([x,y,z],direction_1,direction_2,T,og=og,k_B=k_B,spin=spin,eta=eta,Ef=Ef)

    def spin_hall_conductivity_new(self,k_mesh,dir_1,dir_2,T,og=0,k_B=0.025852,spin=0,eta=1e-3,Ef=0):
        func=functools.partial(self.berry_curvature_onek_forparallel,direction_1=dir_1,direction_2=dir_2,T=T,og=og,k_B=k_B,spin=spin,eta=eta,Ef=Ef)
        if len(k_mesh)!=self._dim_k:
            raise Exception("Wrong, the dimension of k_mesh must equal to dim_k")
        sigma=0
        err=0
        k_range=[]
        if self._dim_k==1:
            nk=k_mesh[0]
            for i in range(nk-1):
                k_min=i/(nk-1)
                k_max=(i+1)/(nk-1)
                k_range.append([[k_min,k_max],[0,1],[0,1]])
        elif self._dim_k==2:
            for i in range(k_mesh[0]-1):
                x_min=i/(k_mesh[0]-1)
                x_max=(i+1)/(k_mesh[0]-1)
                for j in range(k_mesh[1]-1):
                    y_min=j/(k_mesh[1]-1)
                    y_max=(j+1)/(k_mesh[1]-1)
                    k_range.append([[x_min,x_max],[y_min,y_max],[0,1]])
        elif self._dim_k==3:
            for i in range(k_mesh[0]-1):
                x_min=i/(k_mesh[0]-1)
                x_max=(i+1)/(k_mesh[0]-1)
                for j in range(k_mesh[1]-1):
                    y_min=i/(k_mesh[1]-1)
                    y_max=(i+1)/(k_mesh[1]-1)
                    for k in range(k_mesh[2]-1):
                        z_min=k/(k_mesh[1]-1)
                        z_max=(k+1)/(k_mesh[1]-1)
                        k_range.append([[x_min,x_max],[y_min,y_max],[z_min,z_max]])
        k_range=np.array(k_range)
        integ=functools.partial(integrate.nquad,func,opts=dict(epsabs=1e-4,epsrel=1e-4))
        result=[]
        pool=Pool()
        result=pool.map(integ,k_range)
        pool.close()
        pool.join()
        result=np.array(result)
        result=np.sum(result,axis=0)
        sigma=result[0]
        err=result[1]
        print("err=%f"%err)
        sigma*=(2*np.pi)**self._dim_k
        return sigma

class CPA(object):
    def __init__(self,model_1,model_2,ratio):
        self._ratio=ratio
        self._dim_r=model_1._dim_r
        self._dim_k=model_1._dim_k
        c=self._ratio
        use_model_1=copy.deepcopy(model_1)
        use_model_2=copy.deepcopy(model_2)
        self._natom=np.copy(model_1._natom)
        self._atom=np.copy(model_1._atom)
        self._atom_position=np.copy(model_1._atom_position)
        self._nsta=np.copy(model_1._nsta)
        self._norb=np.copy(model_1._norb)
        self._eps_1=np.zeros((self._natom,self._nsta,self._nsta),dtype=complex)
        self._eps_2=np.zeros((self._natom,self._nsta,self._nsta),dtype=complex)
        self._index=np.zeros((self._natom,self._nsta,self._nsta),dtype=bool)
        self._eps=np.zeros((self._natom,self._nsta,self._nsta),dtype=complex)
        self._nspin=model_1._nspin
        atom_a=int(0)
        for i in range(self._natom):
            index=np.zeros((self._nsta,self._nsta),dtype=bool)
            index[atom_a:atom_a+self._atom[i],atom_a:atom_a+self._atom[i]]=True
            if self._nspin==2:
                index[self._norb+atom_a:self._norb+atom_a+self._atom[i],atom_a:atom_a+self._atom[i]]=True
                index[atom_a:atom_a+self._atom[i],self._norb+atom_a:self._norb+atom_a+self._atom[i]]=True
                index[self._norb+atom_a:self._norb+atom_a+self._atom[i],self._norb+atom_a:self._norb+atom_a+self._atom[i]]=True
            self._index[i]=index
            self._eps_1[i]=use_model_1._ham[0]*index 
            self._eps_2[i]=use_model_2._ham[0]*index 
            self._eps[i]=self._eps_1[i]*(1-c)+self._eps_2[i]*c
            use_model_1._ham[0]-=self._eps_1[i]
            use_model_2._ham[0]-=self._eps_2[i]
            atom_a+=self._atom[i]
        self._model=plus(use_model_1,use_model_2,c)



    def cpa_solve(self,energy,kvec,eta=0.01,gen_model=False,nk0=12):
        c=self._ratio
        nk=len(kvec)
        ne=len(energy)
        k_f=k_mesh(nk0,self._dim_k)
        pool=Pool()
        tau=pool.map(self._model.gen_ham,k_f)
        pool.close()
        pool.join()
        tau=np.array(tau)
        energy_window=np.zeros((ne,nk0**self._dim_k,self._nsta,self._nsta),dtype=float)
        energy_window[:,:]=np.identity(self._nsta,dtype=float)
        energy_window=energy[:,None,None,None]*energy_window
        eps_c=np.zeros((ne,self._natom,self._nsta,self._nsta),dtype=complex)
        eps_c[:]=self._eps
        Gk=np.linalg.inv(energy_window+(1.j*eta*np.identity(self._nsta)-np.sum(self._eps,axis=0))[None,None]-tau[None])
        tau=0
        energy_window=0
        del tau 
        del energy_window
        G0=np.zeros((self._natom,ne,self._nsta,self._nsta),dtype=complex)
        G0[:]=1/nk0**self._dim_k*np.sum(Gk,axis=1)
        Gk=0
        del Gk
        G0=G0.transpose(1,0,2,3)
        for i in range(10):
            dt=np.copy(np.sum(eps_c,axis=1))
            G_mcl=np.linalg.inv(np.linalg.inv(G0)+eps_c)
            G1=np.linalg.inv(eps_c-self._eps_1[None,:,:,:]+np.linalg.inv(G0))
            G2=np.linalg.inv(eps_c-self._eps_2[None,:,:,:]+np.linalg.inv(G0))
            G0=(1-c)*G1+c*G2
            eps_c=np.linalg.inv(G_mcl)-np.linalg.inv(G0)
            if np.all(np.sum(eps_c,axis=1)-dt<1e-13):
                print(i)
                break
        tau0=self._model.gen_ham(kvec)
        energy_window=np.zeros((ne,nk,self._nsta,self._nsta),dtype=float)
        energy_window[:,:]=np.identity(self._nsta,dtype=float)
        energy_window=energy[:,None,None,None]*energy_window
        Gk=np.linalg.inv(energy_window+(1.j*eta*np.identity(self._nsta))[None,None]-np.sum(eps_c,axis=1)[:,None,:,:]-tau0[None])
        B=-1/np.pi*np.imag(np.trace(Gk,axis1=2,axis2=3))[::-1]
        B=(B-np.min(B))/(np.max(B)-np.min(B))
        del tau0 
        del Gk 
        del energy_window
        if gen_model:
            self._model._ham[0]+=np.sum(np.sum(eps_c,axis=0)/ne,axis=0)
            use_model_c=copy.deepcopy(self._model)
            return (B,use_model_c)
        else:
            return B

    def cpa_solve_order(self,energy,kvec,eta=0.01,gen_model=False,nk0=12):
        c=self._ratio
        nk=len(kvec)
        ne=len(energy)
        k_f=k_mesh(nk0,self._dim_k)
        pool=Pool()
        tau=pool.map(self._model.gen_ham,k_f)
        pool.close()
        pool.join()
        tau=np.array(tau)
        energy_window=np.zeros((ne,nk0**self._dim_k,self._nsta,self._nsta),dtype=float)
        energy_window[:,:]=np.identity(self._nsta,dtype=float)
        energy_window=energy[:,None,None,None]*energy_window
        eps_c=np.zeros((ne,self._nsta,self._nsta),dtype=complex)
        eps_c[:]=np.sum(self._eps,axis=0)
        Gk=np.linalg.inv(energy_window+(1.j*eta*np.identity(self._nsta)-np.sum(self._eps,axis=0))[None,None]-tau[None])
        tau=0
        energy_window=0
        del tau 
        del energy_window
        G0=np.zeros((ne,self._nsta,self._nsta),dtype=complex)
        G0=1/nk0**self._dim_k*np.sum(Gk,axis=1)
        Gk=0
        del Gk
        epsilon1=self._eps_1[0]+self._eps_1[1]
        epsilon2=self._eps_1[0]+self._eps_2[1]
        epsilon3=self._eps_2[0]+self._eps_1[1]
        epsilon4=self._eps_2[0]+self._eps_2[1]
        P_1 = (1-c)*(1-c)
        P_2 = (1-c)*c
        P_3 = c*(1-c)
        P_4 = c*c
        for i in range(100):
            dt=np.copy(eps_c)
            G_mcl=np.linalg.inv(np.linalg.inv(G0)+eps_c)
            G1=np.linalg.inv(eps_c-epsilon1+np.linalg.inv(G0))
            G2=np.linalg.inv(eps_c-epsilon2+np.linalg.inv(G0))
            G3=np.linalg.inv(eps_c-epsilon3+np.linalg.inv(G0))
            G4=np.linalg.inv(eps_c-epsilon4+np.linalg.inv(G0))
            G0=G1*P_1 + G2*P_2 + G3*P_3 + G4*P_4
            eps_c=np.linalg.inv(G_mcl)-np.linalg.inv(G0)
            print(i)
            if np.all(eps_c-dt<1e-8):
                break
        tau0=self._model.gen_ham(kvec)
        energy_window=np.zeros((ne,nk,self._nsta,self._nsta),dtype=float)
        energy_window[:,:]=np.identity(self._nsta,dtype=float)
        energy_window=energy[:,None,None,None]*energy_window
        Gk=np.linalg.inv(energy_window+(1.j*eta*np.identity(self._nsta))[None,None]-eps_c[:,None,:,:]-tau0[None])
        B=-1/np.pi*np.imag(np.trace(Gk,axis1=2,axis2=3))[::-1]
        B=(B-np.min(B))/(np.max(B)-np.min(B))
        del tau0 
        del Gk 
        del energy_window
        if gen_model:
            self._model._ham[0]+=np.sum(eps_c,axis=0)/ne
            use_model_c=copy.deepcopy(self._model)
            return (B,use_model_c)
        else:
            return B


class w90(object):
    def __init__(self,path,prefix,mode=0,read_r=False):
        self._path=path
        self._prefix=prefix
        f=open(self._path+"/"+self._prefix+".win","r")
        self._win=f.read()
        f.close()
        lat=re.findall("begin unit_cell_cart[\-\. 0-9\n]*end unit_cell_cart",self._win)
        lat=re.findall("[\-\. 0-9]*[0-9]",lat[0])
        spin=re.findall('spinors[\ ]*=[a-z A-Z\.]*',self._win)
        atom=re.findall("begin atoms_cart[A-Za-z\-\. 0-9\n]*end atoms_cart",self._win)
        atom=re.findall("[A-Z][a-z]*",atom[0])
        proj=re.findall("begin projections[A-Za-z;:, \n]*end projections",self._win)
        proj=re.findall("[A-Z][a-z]*:[a-z,;]*",proj[0])
        self._atom=[]
        proj_name=[]
        proj_num=[]
        for i in range(len(proj)):
            proj_name.append(re.findall("[A-Z][a-z]*",proj[i])[0])
            projection=re.findall(":[a-z,;]*",proj[i])
            projection=re.findall("[a-z][a-z]*",projection[0])
            proj_num.append(0)
            for j in range(len(projection)):
                if "s"==projection[j]:
                    proj_num[i]+=1
                elif "px"==projection[j]:
                    proj_num[i]+=1
                elif "py"==projection[j]:
                    proj_num[i]+=1
                elif "py"==projection[j]:
                    proj_num[i]+=1
                elif "p"==projection[j]:
                    proj_num[i]+=3
                elif "sp2"==projection[j]:
                    proj_num[i]+=3
                elif "sp3"==projection[j]:
                    proj_num[i]+=4
                elif "d"==projection[j]:
                    proj_num[i]+=5
                elif "f"==projection[j]:
                    proj_num[i]+=5
                else:
                    raise Exception("cant recognize the projections, the procedure only can recognize s,px,py,pz,p,d,f,sp2,sp3")
        for a,i in enumerate(proj_name):
            for j in atom:
                if j==i:
                    self._atom.append(proj_num[a])
        self._atom=np.array(self._atom,dtype=int)
        self._natom=int(len(self._atom))
        if len(spin)==0:
            self._nspin=1
        else:
            if re.findall('[tT]',spin[0])!=None:
                self._nspin=2
            else:
                self._nspin=1
        self._lat=[]
        for i in lat:
            lats=re.findall("[\-\.0-9]*[0-9]",i)
            latss=[]
            for j in lats:
                latss.append(float(j))
            self._lat.append(latss)
        self._lat=np.array(self._lat,dtype=float)
        f=open(self._path+'/'+prefix+'_hr.dat')
        f.readline()
        n_orb=int(f.readline())
        self._norb=int(n_orb/self._nspin)
        self._nsta=n_orb
        n_R=int(f.readline())
        n_cal=0
        weight=[]
        n_raw=3
        while n_cal<n_R:
            weight=np.append(weight,np.array(re.findall('[0-9]',f.readline()),dtype=int))
            n_cal=len(weight)
            n_raw+=1
        n_line=n_R*n_orb**2
        tmp_data=np.zeros((n_line,2),dtype=float)
        ind_R_data=np.zeros((n_line,3),dtype=int)
        ind_ij=np.zeros((n_line,2),dtype=int)
        gen_data=np.loadtxt(self._path+'/'+prefix+'_hr.dat',dtype=float,skiprows=n_raw)
        tmp=gen_data[:,5]+1.j*gen_data[:,6]
        ind_R_data=np.array(gen_data[:,:3],dtype=int)
        ind_ij=np.array(gen_data[:,3:5],dtype=int)
        del gen_data
        ham=np.zeros((n_R,n_orb,n_orb),dtype=complex)
        R_ham=np.zeros((n_R,3),dtype=int)
        R_ham=ind_R_data[::n_orb**2]
        ham=gen_w90_ham(n_R,n_orb,tmp,weight,ham)
        new_ham=np.copy(ham)
        if mode==1:
            new_ham[:,:self._norb,:self._norb]=ham[:,::2,::2]
            new_ham[:,self._norb:,self._norb:]=ham[:,1::2,1::2]
            new_ham[:,:self._norb,self._norb:]=ham[:,::2,1::2]
            new_ham[:,self._norb:,:self._norb]=ham[:,1::2,::2]
        ham=new_ham
        self._ham=ham
        self._hamR=R_ham
        f=open(self._path+"/"+prefix+"_centres.xyz","r")
        f.readline()
        f.readline()
        self._orb=[]
        self._atom_position=np.zeros((self._natom,3),dtype=float)
        for i in range(self._norb):
            lat=np.array(re.findall("[\-\.0-9]*[0-9]",f.readline()),dtype=float)
            self._orb.append(lat)
        if self._nspin==2:
            for i in range(self._norb):
                f.readline()
        for i in range(self._natom):
            aa=f.readline()
            print(aa)
            lat=np.array(re.findall("[\-\.0-9]*[0-9]",aa),dtype=float)
            self._atom_position[i]=lat
        self._atom_position=shift_to_zero(np.dot(self._atom_position,np.linalg.inv(self._lat)))
        self._orb=np.array(self._orb)
        self._orb=shift_to_zero(np.dot(self._orb,np.linalg.inv(self._lat)))
        self._ef=0.0
        self._per=[0,1,2]
        self._dim_r=3
        self._dim_k=3
        self._natom=len(self._atom)

        #有时候, wannier_centres 的轨道和原子位置不一致, 所以我们需要将其顺序调整到一致, 为了方便, 我们调整原子位置
        index=np.arange(self._natom)
        for i,atom in enumerate(self._atom_position):
            index[i]=np.argmin(np.linalg.norm(atom-self._orb,axis=1))
        index=np.argsort(index)
        self._atom_position=self._atom_position[index]
        self._atom=self._atom[index]
        self._read_r=read_r
        if self._read_r:
            gen_data=np.loadtxt(self._path+'/'+prefix+'_r.dat',dtype=float,skiprows=3)
            R=gen_data[:,:3]
            ind=gen_data[:,3:5]
            rmatrix=np.zeros((3,n_R,n_orb,n_orb),dtype=complex)
            tmp=np.array([gen_data[:,5]+1.j*gen_data[:,6],gen_data[:,7]+1.j*gen_data[:,8],gen_data[:,7]+1.j*gen_data[:,8]],dtype=complex).T
            self._rmatrix=gen_w90_r(n_R,n_orb,tmp,rmatrix)
        else:
            self._rmatrix=None

        
    def model(self,zero_energy=0.0,min_hop=0.0):
        self._ef=zero_energy
        tb=tb_model(3,3,self._lat,self._orb,nspin=self._nspin,atom_list=self._atom,atom_position=self._atom_position)
        if np.any(self._hamR[0]!=0): #为了方便计算, 默认 hamR的第一个一定是[0,0,0]位置.
            index=np.argwhere(np.all(self._hamR==0,axis=1))[0,0]
            ins=np.copy(self._ham[index])
            ind=np.copy(self._hamR[0])
            self._ham[index]=np.copy(self._ham[0])
            self._hamR[index]=ind
            self._ham[0]=ins
            self._hamR[0]*=0
        diag=np.identity(tb._nsta,dtype=complex)*zero_energy
        tb_hamR=self._hamR
        tb_ham=self._ham
        index=(tb_hamR[:,2]>0)+(tb_hamR[:,2]==0)*((tb_hamR[:,1]>0)+(tb_hamR[:,1]==0)*(tb_hamR[:,0]>=0))
        tb._ham=self._ham[index]
        tb._hamR=self._hamR[index]
        if self._read_r:
            tb._rmatrix=self._rmatrix[index]
        tb._ham[0]-=diag
        self._ham[0]-=diag
        if min_hop!=0:
            index=(tb._ham>min_hop).any(axis=1).any(axis=2)
            tb._hamR=tb._hamR[index]
            tb._ham=tb._ham[index]
            tb._rmatrix=tb._rmatrix[index]
        return tb

class wf_array(object):
    r"""

    This class is used to solve a tight-binding model
    :class:`pythtb.tb_model` on a regular or non-regular grid
    of points in reciprocal space and/or parameter space, and
    perform on it various calculations. For example it can be
    used to calculate the Berry phase, Berry curvature, 1st Chern
    number, etc.

    *Regular k-space grid*:
    If the grid is a regular k-mesh (no parametric dimensions),
    a single call to the function
    :func:`pythtb.wf_array.solve_on_grid` will both construct a
    k-mesh that uniformly covers the Brillouin zone, and populate
    it with wavefunctions (eigenvectors) computed on this grid.
    The last point in each k-dimension is set so that it represents
    the same Bloch function as the first one (this involves the
    insertion of some orbital-position-dependent phase factors).

    Example :ref:`haldane_bp-example` shows how to use wf_array on
    a regular grid of points in k-space. Examples :ref:`cone-example`
    and :ref:`3site_cycle-example` show how to use non-regular grid of
    points.

    *Parametric or irregular k-space grid grid*:
    An irregular grid of points, or a grid that includes also
    one or more parametric dimensions, can be populated manually
    with the help of the *[]* operator.  For example, to copy
    eigenvectors *evec* into coordinate (2,3) in the *wf_array*
    object *wf* one can simply do::

      wf[2,3]=evec

    The eigenvectors (wavefunctions) *evec* in the example above
    are expected to be in the format *evec[band,orbital]*
    (or *evec[band,orbital,spin]* for the spinfull calculation).
    This is the same format as returned by
    :func:`pythtb.tb_model.solve_one` or
    :func:`pythtb.tb_model.solve_all` (in the latter case one
    needs to restrict it to a single k-point as *evec[:,kpt,:]*
    if the model has *dim_k>=1*).

    If wf_array is used for closed paths, either in a
    reciprocal-space or parametric direction, then one needs to
    include both the starting and ending eigenfunctions even though
    they are physically equivalent.  If the array dimension in
    question is a k-vector direction and the path traverses the
    Brillouin zone in a primitive reciprocal-lattice direction,
    :func:`pythtb.wf_array.impose_pbc` can be used to associate
    the starting and ending points with each other; if it is a
    non-winding loop in k-space or a loop in parameter space,
    then :func:`pythtb.wf_array.impose_loop` can be used instead.
    (These may not be necessary if only Berry fluxes are needed.)

    Example :ref:`3site_cycle-example` shows how one
    of the directions of *wf_array* object need not be a k-vector
    direction, but can instead be a Hamiltonian parameter :math:`\lambda`
    (see also discussion after equation 4.1 in :download:`notes on
    tight-binding formalism <misc/pythtb-formalism.pdf>`).

    :param model: Object of type :class:`pythtb.tb_model` representing
      tight-binding model associated with this array of eigenvectors.

    :param mesh_arr: Array giving a dimension of the grid of points in
      each reciprocal-space or parametric direction.

    Example usage::

      # Construct wf_array capable of storing an 11x21 array of
      # wavefunctions      
      wf = wf_array(tb, [11, 21])
      # populate this wf_array with regular grid of points in
      # Brillouin zone
      wf.solve_on_grid([0.0, 0.0])
      
      # Compute set of eigenvectors at one k-point
      (eval, evec) = tb.solve_one([kx, ky], eig_vectors = True)
      # Store it manually into a specified location in the array
      wf[3, 4] = evec
      # To access the eigenvectors from the same position
      print wf[3, 4]

    """
    def __init__(self,model,mesh_arr):
        # number of electronic states for each k-point
        self._nsta=model._nsta
        # number of spin components
        self._nspin=model._nspin
        # number of orbitals
        self._norb=model._norb
        # store orbitals from the model
        self._orb=np.copy(model._orb)
        # store entire model as well
        self._model=copy.deepcopy(model)
        # store dimension of array of points on which to keep wavefunctions
        self._mesh_arr=np.array(mesh_arr)        
        self._dim_r=model._dim_r
        self._dim_k=model._dim_k
        self._dim_arr=len(self._mesh_arr)
        # all dimensions should be 2 or larger, because pbc can be used
        if True in (self._mesh_arr<=1).tolist():
            raise Exception("\n\nDimension of wf_array object in each direction must be 2 or larger.")
        # generate temporary array used later to generate object ._wfs
        wfs_dim=np.copy(self._mesh_arr)
        wfs_dim=np.append(wfs_dim,self._nsta)
        self._band=np.zeros(wfs_dim,dtype=float)
        wfs_dim=np.append(wfs_dim,self._nsta)
        # store wavefunctions here in the form _wfs[kx_index,ky_index, ... ,band,orb,spin]
        self._wfs=np.zeros(wfs_dim,dtype=complex)

    def solve_on_grid(self,start_k):
        r"""

        Solve a tight-binding model on a regular mesh of k-points covering
        the entire reciprocal-space unit cell. Both points at the opposite
        sides of reciprocal-space unit cell are included in the array.

        This function also automatically imposes periodic boundary
        conditions on the eigenfunctions. See also the discussion in
        :func:`pythtb.wf_array.impose_pbc`.

        :param start_k: Origin of a regular grid of points in the reciprocal space.

        :returns:
          * **gaps** -- returns minimal direct bandgap between n-th and n+1-th 
              band on all the k-points in the mesh.  Note that in the case of band
              crossings one may have to use very dense k-meshes to resolve
              the crossing.

        Example usage::

          # Solve eigenvectors on a regular grid anchored
          # at a given point
          wf.solve_on_grid([-0.5, -0.5])

        """
        solve=functools.partial(self._model.solve_one,eig_vectors=True)
        # check dimensionality
        if self._dim_arr!=self._model._dim_k:
            raise Exception("\n\nIf using solve_on_grid method, dimension of wf_array must equal dim_k of the tight-binding model!")
        # to return gaps at all k-points

        if self._dim_arr==1:
            # don't need to go over the last point because that will be
            # computed in the impose_pbc call
            for i in range(self._mesh_arr[0]-1):
                # generate a kpoint
                kpt=[start_k[0]+float(i)/float(self._mesh_arr[0]-1)]
                # solve at that point
                (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
                # store wavefunctions
                self[i]=evec
                # store gaps
            # impose boundary conditions
            self.impose_pbc(0,self._model._per[0])
        elif self._dim_arr==2:
            kpt=[]
            index=[]
            kpt=gen_mesh_arr(self._mesh_arr,self._dim_arr,start_k)
            (eval,evec)=self._model.solve_all_parallel(kpt,eig_vectors=True)
            wfs_shape=np.array(self._wfs.shape)
            band_shape=np.array(self._band.shape)
            wfs_shape[[0,1]]-=1
            band_shape[[0,1]]-=1
            self._wfs[:-1,:-1]=np.reshape(evec,wfs_shape)
            self._band[:-1,:-1]=np.reshape(eval,band_shape)
            for dir in range(2):
                self.impose_pbc(dir,self._model._per[dir])
        elif self._dim_arr==3:
            kpt=[]
            index=[]
            kpt=gen_mesh_arr(self._mesh_arr,self._dim_arr,start_k)
            (eval,evec)=self._model.solve_all_parallel(kpt,eig_vectors=True)
            wfs_shape=np.array(self._wfs.shape)
            band_shape=np.array(self._band.shape)
            wfs_shape[[0,1,2]]-=1
            band_shape[[0,1,2]]-=1
            self._wfs[:-1,:-1,:-1]=np.reshape(evec,wfs_shape)
            self._band[:-1,:-1,:-1]=np.reshape(eval,band_shape)
            for dir in range(3):
                self.impose_pbc(dir,self._model._per[dir])
        elif self._dim_arr==4:
            kpt=gen_mesh_arr(self._mesh_arr,self._dim_arr,start_k)
            (eval,evec)=self._model.solve_all_parallel(kpt,eig_vectors=True)
            wfs_shape=np.array(self._wfs.shape)
            band_shape=np.array(self._band.shape)
            wfs_shape[[0,1,2,3]]-=1
            band_shape[[0,1,2,3]]-=1
            self._wfs[:-1,:-1,:-1,:-1]=np.reshape(evec,wfs_shape)
            self._band[:-1,:-1,:-1,:-1]=np.reshape(eval,band_shape)
            for dir in range(4):
                self.impose_pbc(dir,self._model._per[dir])
        else:
            raise Exception("\n\nWrong dimensionality!")

    def __check_key(self,key):
        # do some checks for 1D
        if self._dim_arr==1:
            if type(key).__name__!='int':
                raise TypeError("Key should be an integer!")
            if key<(-1)*self._mesh_arr[0] or key>=self._mesh_arr[0]:
                raise IndexError("Key outside the range!")
        # do checks for higher dimension
        else:
            if len(key)!=self._dim_arr:
                raise TypeError("Wrong dimensionality of key!")
            for i,k in enumerate(key):
                if type(k).__name__!='int':
                    raise TypeError("Key should be set of integers!")
                if k<(-1)*self._mesh_arr[i] or k>=self._mesh_arr[i]:
                    raise IndexError("Key outside the range!")

    def __getitem__(self,key):
        # check that key is in the correct range
        self.__check_key(key)
        # return wavefunction
        return self._wfs[key]
    
    def __setitem__(self,key,value):
        # check that key is in the correct range
        self.__check_key(key)
        # store wavefunction
        self._wfs[key]=np.array(value,dtype=complex)

    def impose_pbc(self,mesh_dir,k_dir):
        r"""

        If the *wf_array* object was populated using the
        :func:`pythtb.wf_array.solve_on_grid` method, this function
        should not be used since it will be called automatically by
        the code.

        The eigenfunctions :math:`\Psi_{n {\bf k}}` are by convention
        chosen to obey a periodic gauge, i.e.,
        :math:`\Psi_{n,{\bf k+G}}=\Psi_{n {\bf k}}` not only up to a
        phase, but they are also equal in phase.  It follows that
        the cell-periodic Bloch functions are related by
        :math:`u_{n,{\bf k+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k}}`.
        See :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>` section 4.4 and equation 4.18 for
        more detail.  This routine sets the cell-periodic Bloch function
        at the end of the string in direction :math:`{\bf G}` according
        to this formula, overwriting the previous value.

        This function will impose these periodic boundary conditions along
        one direction of the array. We are assuming that the k-point
        mesh increases by exactly one reciprocal lattice vector along
        this direction. This is currently **not** checked by the code;
        it is the responsibility of the user. Currently *wf_array*
        does not store the k-vectors on which the model was solved;
        it only stores the eigenvectors (wavefunctions).
        
        :param mesh_dir: Direction of wf_array along which you wish to
          impose periodic boundary conditions.

        :param k_dir: Corresponding to the periodic k-vector direction
          in the Brillouin zone of the underlying *tb_model*.  Since
          version 1.7.0 this parameter is defined so that it is
          specified between 0 and *dim_r-1*.

        See example :ref:`3site_cycle-example`, where the periodic boundary
        condition is applied only along one direction of *wf_array*.

        Example usage::

          # Imposes periodic boundary conditions along the mesh_dir=0
          # direction of the wf_array object, assuming that along that
          # direction the k_dir=1 component of the k-vector is increased
          # by one reciprocal lattice vector.  This could happen, for
          # example, if the underlying tb_model is two dimensional but
          # wf_array is a one-dimensional path along k_y direction.          
          wf.impose_pbc(mesh_dir=0,k_dir=1)

        """

        if k_dir not in self._model._per:
            raise Exception("Periodic boundary condition can be specified only along periodic directions!")

        # Compute phase factors
        ffac=np.exp(-2.j*np.pi*self._orb[:,k_dir])
        if self._nspin==1:
            phase=ffac
        else:
            # for spinors, same phase multiplies both components
            phase=np.zeros(self._nsta,dtype=complex)
            phase[:self._norb]=ffac
            phase[self._norb:]=ffac
        
        # Copy first eigenvector onto last one, multiplying by phase factors
        # We can use numpy broadcasting since the orbital index is last
        if mesh_dir==0:
            self._wfs[-1,...]=self._wfs[0,...]*phase
        elif mesh_dir==1:
            self._wfs[:,-1,...]=self._wfs[:,0,...]*phase
        elif mesh_dir==2:
            self._wfs[:,:,-1,...]=self._wfs[:,:,0,...]*phase
        elif mesh_dir==3:
            self._wfs[:,:,:,-1,...]=self._wfs[:,:,:,0,...]*phase
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

    def impose_loop(self,mesh_dir):
        r"""

        If the user knows that the first and last points along the
        *mesh_dir* direction correspond to the same Hamiltonian (this
        is **not** checked), then this routine can be used to set the
        eigenvectors equal (with equal phase), by replacing the last
        eigenvector with the first one (for each band, and for each
        other mesh direction, if any).

        This routine should not be used if the first and last points
        are related by a reciprocal lattice vector; in that case,
        :func:`pythtb.wf_array.impose_pbc` should be used instead.

        :param mesh_dir: Direction of wf_array along which you wish to
          impose periodic boundary conditions.

        Example usage::

          # Suppose the wf_array object is three-dimensional
          # corresponding to (kx,ky,lambda) where (kx,ky) are
          # wavevectors of a 2D insulator and lambda is an
          # adiabatic parameter that goes around a closed loop.
          # Then to insure that the states at the ends of the lambda
          # path are equal (with equal phase) in preparation for
          # computing Berry phases in lambda for given (kx,ky),
          # do wf.impose_loop(mesh_dir=2)

        """

        # Copy first eigenvector onto last one
        if mesh_dir==0:
            self._wfs[-1,...]=self._wfs[0,...]
        elif mesh_dir==1:
            self._wfs[:,-1,...]=self._wfs[:,0,...]
        elif mesh_dir==2:
            self._wfs[:,:,-1,...]=self._wfs[:,:,0,...]
        elif mesh_dir==3:
            self._wfs[:,:,:,-1,...]=self._wfs[:,:,:,0,...]
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

    def Wilson_ham(self,occ,dir=None,berry_evals=False):
            loop=self.Wilson_loop(occ,dir,berry_evals)
            mesh_arr=np.delete(self._mesh_arr,dir)
            ham=np.zeros_like(loop,dtype=complex)
            if self._dim_arr==2:
              for i in range(mesh_arr[0]):
                ham[i,...]=1.j*linalg.logm(loop[i,...])
            elif self._dim_arr==3:
              for i in range(mesh_arr[0]):
                for j in range(mesh_arr[1]):
                  ham[i,j,...]=1.j*linalg.logm(loop[i,j,...])
            return ham


    def Wilson_loop(self,occ,dir=None,berry_evals=False):
            #if dir<0 or dir>self._dim_arr-1:
            #  raise Exception("\n\nDirection key out of range")
            #
            # This could be coded more efficiently, but it is hard-coded for now.
            #
            # 1D case
            if self._dim_arr==1:
                # pick which wavefunctions to use
                wf_use=self._wfs[:,occ,:]
                # calculate berry phase
                ret=__wilson_loop__(wf_use,berry_evals)
            # 2D case
            elif self._dim_arr==2:
                # choice along which direction you wish to calculate berry phase
                if dir==0:
                    ret=[]
                    for i in range(self._mesh_arr[1]):
                        wf_use=self._wfs[:,i,:,:][:,occ,:]
                        ret.append(__wilson_loop__(wf_use,berry_evals))
                elif dir==1:
                    ret=[]
                    for i in range(self._mesh_arr[0]):
                        wf_use=self._wfs[i,:,:,:][:,occ,:]
                        ret.append(__wilson_loop__(wf_use,berry_evals))
                else:
                    raise Exception("\n\nWrong direction for Berry phase calculation!")
            # 3D case
            elif self._dim_arr==3:
                # choice along which direction you wish to calculate berry phase
                if dir==0:
                    ret=[]
                     
                    for i in range(self._mesh_arr[1]):
                        ret_t=[]
                        for j in range(self._mesh_arr[2]):
                            wf_use=self._wfs[:,i,j,:,:][:,occ,:]
                            ret_t.append(__wilson_loop__(wf_use,berry_evals))
                        ret.append(ret_t)
                elif dir==1:
                    ret=[]
                    for i in range(self._mesh_arr[0]):
                        ret_t=[]
                        for j in range(self._mesh_arr[2]):
                            wf_use=self._wfs[i,:,j,:,:][:,occ,:]
                            ret_t.append(__wilson_loop__(wf_use,berry_evals))
                        ret.append(ret_t)
                elif dir==2:
                    ret=[]
                    for i in range(self._mesh_arr[0]):
                        ret_t=[]
                        for j in range(self._mesh_arr[1]):
                            wf_use=self._wfs[i,j,:,:,:][:,occ,:]
                            ret_t.append(__wilson_loop__(wf_use,berry_evals))
                        ret.append(ret_t)
                else:
                    raise Exception("\n\nWrong direction for Berry phase calculation!")
            else:
                raise Exception("\n\nWrong dimensionality!")

            # convert phases to numpy array
            if self._dim_arr>1 or berry_evals==True:
                ret=np.array(ret,dtype=complex)
            return ret


    def berry_phase(self,occ,dir=None,contin=True,berry_evals=False):
        #if dir<0 or dir>self._dim_arr-1:
        #  raise Exception("\n\nDirection key out of range")
        #
        # This could be coded more efficiently, but it is hard-coded for now.
        #
        # 1D case
        if self._dim_arr==1:
            # pick which wavefunctions to use
            wf_use=self._wfs[:,occ,:]
            # calculate berry phase
            ret=_one_berry_loop(wf_use,berry_evals)
        # 2D case
        elif self._dim_arr==2:
            # choice along which direction you wish to calculate berry phase
            if dir==0:
                ret=[]
                for i in range(self._mesh_arr[1]):
                    wf_use=self._wfs[:,i,:,:][:,occ,:]
                    ret.append(_one_berry_loop(wf_use,berry_evals))
            elif dir==1:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    wf_use=self._wfs[i,:,:,:][:,occ,:]
                    ret.append(_one_berry_loop(wf_use,berry_evals))
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        # 3D case
        elif self._dim_arr==3:
            # choice along which direction you wish to calculate berry phase
            if dir==0:
                ret=[]
                 
                for i in range(self._mesh_arr[1]):
                    ret_t=[]
                    for j in range(self._mesh_arr[2]):
                        wf_use=self._wfs[:,i,j,:,:][:,occ,:]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            elif dir==1:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    ret_t=[]
                    for j in range(self._mesh_arr[2]):
                        wf_use=self._wfs[i,:,j,:,:][:,occ,:]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            elif dir==2:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    ret_t=[]
                    for j in range(self._mesh_arr[1]):
                        wf_use=self._wfs[i,j,:,:,:][:,occ,:]
                        ret_t.append(_one_berry_loop(wf_use,berry_evals))
                    ret.append(ret_t)
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        else:
            raise Exception("\n\nWrong dimensionality!")

        # convert phases to numpy array
        if self._dim_arr>1 or berry_evals==True:
            ret=np.array(ret,dtype=float)

        # make phases of eigenvalues continuous
        if contin==True:
            # iron out 2pi jumps, make the gauge choice such that first phase in the
            # list is fixed, others are then made continuous.
            if berry_evals==False:
                # 2D case
                if self._dim_arr==2:
                    ret=_one_phase_cont(ret,ret[0])
                # 3D case
                elif self._dim_arr==3:
                    for i in range(ret.shape[1]):
                        if i==0: clos=ret[0,0]
                        else: clos=ret[0,i-1]
                        ret[:,i]=_one_phase_cont(ret[:,i],clos)
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
            # make eigenvalues continuous. This does not take care of band-character
            # at band crossing for example it will just connect pairs that are closest
            # at neighboring points.
            else:
                # 2D case
                if self._dim_arr==2:
                    ret=_array_phases_cont(ret,ret[0,:])
                # 3D case
                elif self._dim_arr==3:
                    for i in range(ret.shape[1]):
                        if i==0: clos=ret[0,0,:]
                        else: clos=ret[0,i-1,:]
                        ret[:,i]=_array_phases_cont(ret[:,i],clos)
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
        ret.sort()
        return ret

    def berry_flux(self,occ,dirs=None,individual_phases=False):
        if dirs==None:
            dirs=[0,1]

        if dirs[0]==dirs[1]:
            raise Exception("Need to specify two different directions for Berry flux calculation.")
        elif dirs[0]>= self._dim_arr or dirs[1]>=self._dim_arr or dirs[0]<0 or dirs[1]<0:
            raise Exception("Direction for Berry flux calculation out of bounds.")
        
        if self._dim_arr==2:
            ord=list(range(len(self._wfs.shape)))
            ord[0]=dirs[0]
            ord[1]=dirs[1]
            plane_wfs=self._wfs.transpose(ord)
            plane_wfs=plane_wfs[:,:,occ]
            all_phases=_one_flux_plane(plane_wfs)
            if individual_phases==False:
                return all_phases.sum()
            else:
                return all_phases
        elif self._dim_arr in [3,4]:
            ord=list(range(len(self._wfs.shape)))
            ord[0]=dirs[0]
            ord[1]=dirs[1]
            ld=list(range(self._dim_arr))
            ld.remove(dirs[0])
            ld.remove(dirs[1])
            if len(ld)!=self._dim_arr-2:
                raise Exception("Hm, this should not happen? Inconsistency with the mesh size.")
            if self._dim_arr==3:
                ord[2]=ld[0]
            if self._dim_arr==4:
                ord[2]=ld[0]
                ord[3]=ld[1]
            use_wfs=self._wfs.transpose(ord)


            if self._dim_arr==3:
                slice_phases=np.zeros((self._mesharr[ord[2]],self._mesh_arr[dirs[0]]-1,self._mesh_arr[dirs[1]]-1),dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    plane_wfs=plane_wfs[:,:,i,occ]
                    slice_phases[i,:,:]=_one_flux_plane(plane_wfs)
            elif self._dim_arr==4:
                slice_phases=np.zeros((self._mesh_arr[ord[2]],self._mesh_arr[ord[3]],self._mesh_arr[dirs[0]]-1,self._mesh_arr[dirs[1]]-1),dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    for j in range(self._mesh_arr[ord[3]]):
                        plane_wfs=use_wfs[:,:,i,j,occ]
                        slice_phases[i,j,:,:]=_one_flux_plane(plane_wfs)
                if individual_phases==False:
                    return slice_phases.sum(asix=(-2,-1))
                else:
                    return slice_phases

            else:
                raise Exception("\n\n Wrong dimensionality!")


    def dos(self,energy_range,n_e,method='Gaussian',sigma=0.01):
        start_k=np.zeros(self._dim_k,dtype=float)
        n_e=int(n_e)
        E0=np.linspace(energy_range[0],energy_range[1],n_e)
        band=np.copy(self._band)
        for i in range(self._dim_k):
            band=np.delete(band,-1,axis=i)
        evals=band.flatten()
        if 'G' in method:
            center=evals.ravel()
            dos=np.zeros(n_e,dtype=float)
            dos=Gauss(E0,dos,center,sigma)
        return dos



    def optical_conductivity(self,dir_1,dir_2,T,og,eta=0.001,k_B=0.025852):
        r"""
        这个函数是用来计算霍尔电导率的
        """
        start_k=np.zeros(self._dim_k,dtype=float)
        kpt=gen_mesh_arr(self._mesh_arr,self._dim_arr,start_k)
        dir_1=np.array(dir_1)
        dir_2=np.array(dir_2)
        pool=Pool()
        v=pool.map(self._model.gen_v,kpt)
        pool.close()
        pool.join()
        nk=len(kpt)
        V=np.linalg.det(self._model._lat)
        if len(self._mesh_arr)==1:
            band=np.reshape(self._band[:-1],(nk,self._model._nsta))
            evec=np.reshape(self._wfs[:-1],(nk,self._model._nsta,self._model._nsta))
        elif len(self._mesh_arr)==2:
            band=np.reshape(self._band[:-1,:-1],(nk,self._model._nsta))
            evec=np.reshape(self._wfs[:-1,:-1],(nk,self._model._nsta,self._model._nsta))
        elif len(self._mesh_arr)==3:
            band=np.reshape(self._band[:-1,:-1,:-1],(nk,self._model._nsta))
            evec=np.reshape(self._wfs[:-1,:-1,:-1],(nk,self._model._nsta,self._model._nsta))
        v1=np.einsum("krij,r->kij",v,dir_1)
        v2=np.einsum("krij,r->kij",v,dir_2)
        if T>1:
            beta=1/(T*k_B)
            fermi_dirac=1/(np.exp(beta*band)+1)
        else:
            fermi_dirac=np.array(band<=0,dtype=float)
        evec_T=evec.transpose(0,2,1)
        evec_j=evec.conj()
        A1=np.matmul(evec_j,np.matmul(v1,evec_T))
        A2=np.matmul(evec_j,np.matmul(v2,evec_T))
        omega_n=np.zeros((nk,self._model._nsta),dtype=float)
        for i in range(self._model._nsta):
            for j in range(self._model._nsta):
                B=fermi_dirac[:,i]-fermi_dirac[:,j]
                B0=(band[:,i]-band[:,j])
                ind=(np.abs(B0)>1e-5)
                if ind.any():
                    omega_n[:,i]+=-2*(B/B0*A1[:,i,j]*A2[:,j,i]/(band[:,i]-band[:,j]-(og+1.j*eta))).imag
        sigma=np.sum(omega_n)/nk/V*(2*np.pi)**self._model._dim_k
        return sigma 

    def spin_hall_conductivity(self,dir_1,dir_2,T,og,eta=0.001,k_B=0.025852,spin=0):
        start_k=np.zeros(self._dim_k,dtype=float)
        kpt=gen_mesh_arr(self._mesh_arr,self._dim_arr,start_k)
        dir_1=np.array(dir_1)
        dir_2=np.array(dir_2)
        nsta=self._model._nsta
        pool=Pool()
        v=pool.map(self._model.gen_v,kpt)
        pool.close()
        pool.join()
        nk=len(kpt)
        V=np.linalg.det(self._model._lat)
        if len(self._mesh_arr)==1:
            band=np.reshape(self._band[:-1],(nk,self._model._nsta))
            evec=np.reshape(self._wfs[:-1],(nk,self._model._nsta,self._model._nsta))
        elif len(self._mesh_arr)==2:
            band=np.reshape(self._band[:-1,:-1],(nk,self._model._nsta))
            evec=np.reshape(self._wfs[:-1,:-1],(nk,self._model._nsta,self._model._nsta))
        elif len(self._mesh_arr)==3:
            band=np.reshape(self._band[:-1,:-1,:-1],(nk,self._model._nsta))
            evec=np.reshape(self._wfs[:-1,:-1,:-1],(nk,self._model._nsta,self._model._nsta))
        if self._nspin==2:
            if spin==0:
                s=np.array([[1,0],[0,1]],dtype=complex)/2
            elif spin==1:
                s=np.array([[0,1],[1,0]],dtype=complex)/2
            elif spin==2:
                s=np.array([[0,-1.j],[1.j,0]],dtype=complex)/2
            elif spin==3:
                s=np.array([[1,0],[0,-1]],dtype=complex)/2
            X=np.kron(s,np.eye(self._norb))
            J=(np.matmul(X,v)+np.matmul(v,X))/2
        else:
            J=v
        v1=np.einsum("krij,r->kij",J,dir_1)
        v2=np.einsum("krij,r->kij",v,dir_2)
        if T>1:
            beta=1/(T*k_B)
            fermi_dirac=1/(np.exp(beta*band)+1)
        else:
            fermi_dirac=np.array(band<=0,dtype=float)
        A1=evec.conj()@(v1@evec.transpose(0,2,1))
        A2=evec.conj()@(v2@evec.transpose(0,2,1))
        omega_n=np.zeros((nk,self._model._nsta),dtype=float)
        U0=np.zeros((nk,nsta,nsta),dtype=complex)
        for i in range(nsta):
            for j in range(nsta):
                U0[:,i,j]=1/((band[:,i]-band[:,j])**2-(og+1.j*eta)**2)
        omega_n=-2*np.einsum("kij,kji,kij->ki",A1,A2,U0).imag
        omega=np.sum(omega_n*fermi_dirac,axis=1)
        omega=np.reshape(omega,self._mesh_arr-1)
        sigma=np.sum(omega)/nk*(2*np.pi)**self._model._dim_k
        return sigma 
        




class Green_function(object):
    def __init__(self,model,haf_fin_dim=None,Np=None):
        if model._dim_k==0:
            raise Exception("\n\n the model's k dimension must >=1")
        elif model._dim_k==1 and haf_fin_dim!=None:
            raise Exception("\n\n the model is one dimension, you shouldn't give the haf_fin_dim")
        elif (model._dim_k==2 or model._dim_k==3) and haf_fin_dim==None:
            raise Exception("\n\n the model's k dimension is larger than one, you need to give the haf_fin_dim")

        per=copy.copy(model._per)
        per.remove(haf_fin_dim)
        useR=np.copy(model._hamR)
        if Np!=None:
            num=Np
        else:
            num=np.max(np.abs(useR[:,haf_fin_dim]))  #Find the largest distance of hopping
        use_structure=np.ones(model._dim_r,dtype=int)
        use_structure[haf_fin_dim]=num
        use_structure=np.diag(use_structure)
        use_model=model.make_supercell_new(use_structure) #Make the hopping be the near hopping
        self.model=copy.deepcopy(use_model)
        self._nspin=self.model._nspin
        self._per=per
        self._dim_k=use_model._dim_k-1
        ##################give the hopping of the Hamiltonian##########################
        useR=np.array(self.model._hamR)
        useham=np.array(self.model._ham)
        ###############First for the Bulk Hamiltonian#################
        use_ham_R=np.array(useR)[:,haf_fin_dim]==0
        use_ham_RR_1=(np.array(self.model._hamR)[:,haf_fin_dim]==1)
        use_ham_RR_2=(np.array(self.model._hamR)[:,haf_fin_dim]==-1)
        useham[use_ham_RR_2]=useham[use_ham_RR_2].transpose(0,2,1).conjugate()
        useR[use_ham_RR_2]*=-1
        use_ham_RR=np.logical_or(use_ham_RR_1,use_ham_RR_2)
        self._hamR_R=useR[use_ham_R]
        self._hamR_RR=useR[use_ham_RR]
        self._ham_R=useham[use_ham_R]
        self._ham_RR=useham[use_ham_RR]
        self._norb=self.model._norb
        self._nsta=self._norb*self._nspin
        self._orb=self.model._orb

    def _gen_ham(self,k_input):
        k_point=np.array(k_input)
        if len(k_point.shape)==0:
            k_point=np.array([k_point])
        if self._dim_k==0:
            raise Exception("\n\nthe k dimension must larger than one!")
        if len(k_point.shape)==2:
            lists=True
            if k_point.shape[1]!=self._dim_k:
                raise Exception("\n\nk-vector of wrong shape!")
        else:
            lists=False
            if k_point.shape[0]!=self._dim_k:
                raise Exception("\n\nk-vector of wrong shape!")
        use_indR_R=np.array(self._hamR_R,dtype=int)
        use_indR_RR=np.array(self._hamR_RR,dtype=int)
        orb=np.array(self.model._orb)
        if self._dim_k<1:
            raise Exception("Wrong, the dimension must larger than 1")
        if np.any(use_indR_R[0,self._per]!=0):
            index=np.argwhere(np.all(self._hamR_R[0,self._per]==0,axis=1))[0,0]
            if index.shape[0]==0:
                self._ham_R=np.append(self._ham_R,np.zeros((self._nsta,self._nsta),dtype=complex),axis=0)
                self._hamR_R=np.append(self._hamR_R,np.zeros(self._nsta,dtype=int),axis=0)
                index=self._ham_R.shape[0]
            ins=np.copy(self._ham_R[index])
            ind=np.copy(self._hamR_R[0])
            self._ham_R[index]=np.copy(self._ham[0])
            self._hamR_R[index]=ind
            self._ham_R[0]=ins
            self._hamR_R[0]*=0
        usehamR=np.append(self._ham_R,self._ham_R[1:].transpose((0,2,1)).conjugate(),axis=0)
        use_indR_R=np.append(self._hamR_R,-self._hamR_R[1:],axis=0)[:,self._per]
        usehamRR=np.array(self._ham_RR)
        use_indR_RR=np.array(self._hamR_RR)[:,self._per]
        ham_R=np.sum(np.exp(2.j*np.pi*np.dot(use_indR_R,k_point))[:,None,None]*usehamR,axis=0)
        ham_RR=np.sum(np.exp(2.j*np.pi*np.dot(use_indR_RR,k_point))[:,None,None]*usehamRR,axis=0)
        U=np.diag(np.exp(2.j*np.pi*np.dot(orb[:,self._per],k_point)))
        if self._nspin==2:
            U=np.kron([[1,0],[0,1]],U)
        ham_R=np.dot(ham_R,U)
        ham_RR=np.dot(ham_RR,U)
        ham_R=np.dot(U.T.conjugate(),ham_R)
        ham_RR=np.dot(U.T.conjugate(),ham_RR)
        return (ham_R,ham_RR)


    def green_solve_one(self,k_vec,energy,eta=0.05,N_cut=10):
        kpnt=np.array(k_vec)
        if len(kpnt.shape)==0:
            kpnt=np.array([kpnt])
        if len(kpnt)!=self._dim_k:
            raise Exception("the kpnt must equal to k dimension")
        (ham_R,ham_RR)=self._gen_ham(k_vec)
        ham_RR_dag=ham_RR.conjugate().T
        norb=self._norb
        nsta=self._nsta
        I0=np.identity(nsta,dtype=complex)
        accurate=float(1E-8)
        if type(energy) not in (list,range,type(np.array([0]))):
            epsilon=(energy+eta*1.j)*I0
            ap=ham_RR
            bt=ham_RR_dag
            ep=ham_R
            ep_t=ham_R
            ep_i=ham_R
            @njit()
            def solve_green_T2(ap,bt,ep,ep_i,ep_t,epsilon):
                for i in range(N_cut):
                    g0=np.linalg.inv(epsilon-ep_i)
                    mat_1=np.dot(ap,g0)
                    mat_2=np.dot(bt,g0)
                    g0=np.dot(mat_1,bt)
                    ep_i+=g0
                    ep+=g0
                    g0=np.dot(mat_2,ap)
                    ep_i+=g0
                    ep_t+=g0
                    ap=np.dot(mat_1,ap)
                    bt=np.dot(mat_2,bt)
                    if np.sum(np.abs(ap))<accurate:
                        break
                return (ep,ep_s,ep_t)
            (ep,ep_s,ep_t)=solve_green_T2(ap,bt,ep,ep_i,ep_t,epsilon)
            g_LL=np.linalg.inv(epsilon-ep)
            g_RR=np.linalg.inv(epsilon-ep_t)
            g_B=np.linalg.inv(epsilon-ep_i)
            N_R=-1/np.pi*np.imag(np.trace(g_RR))
            N_L=-1/np.pi*np.imag(np.trace(g_LL))
            N_B=-1/np.pi*np.imag(np.trace(g_B))
            N=[N_R,N_L,N_B]
            return N
        else:
            Ne=int(len(energy))
            epsilon=np.einsum('e,ij->eij',(energy+eta*1.j),I0,dtype=complex)
            ap=np.einsum('e,ij->eij',np.ones(Ne),ham_RR,dtype=complex)
            bt=np.einsum('e,ij->eij',np.ones(Ne),ham_RR_dag,dtype=complex)
            ep=np.einsum('e,ij->eij',np.ones(Ne),ham_R,dtype=complex)
            ep_t=np.copy(ep)
            ep_i=np.copy(ep_t)
            for i in range(N_cut):
                g0=np.linalg.inv(epsilon-ep_i)
                mat_1=np.matmul(ap,g0)
                mat_2=np.matmul(bt,g0)
                g0=np.matmul(mat_1,bt)
                ep_i+=g0
                ep+=g0
                g0=np.matmul(mat_2,ap)
                ep_i+=g0
                ep_t+=g0
                ap=np.matmul(mat_1,ap)
                bt=np.matmul(mat_2,bt)
                if np.all(np.sum(np.abs(ap),axis=(1,2))<accurate):
                    break
            g_LL=np.linalg.inv(epsilon-ep)
            g_RR=np.linalg.inv(epsilon-ep_t)
            g_B=np.linalg.inv(epsilon-ep_i)
            N_R=-1/np.pi*np.imag(np.trace(g_RR,axis1=1,axis2=2))
            N_L=-1/np.pi*np.imag(np.trace(g_LL,axis1=1,axis2=2))
            N_B=-1/np.pi*np.imag(np.trace(g_B,axis1=1,axis2=2))
            N=np.array([N_R,N_L,N_B]).T
            return N

    def k_path(self,kpts,nk):

        if kpts=='full':
            # full Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5],[1.]])
        elif kpts=='fullc':
            # centered full Brillouin zone for 1D case
            k_list=np.array([[-0.5],[0.],[0.5]])
        elif kpts=='half':
            # half Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5]])
        else:
            k_list=np.array(kpts)
    
        # in 1D case if path is specified as a vector, convert it to an (n,1) array
        if len(k_list.shape)==1 and self._dim_k==1:
            k_list=np.array([k_list]).T

        # make sure that k-points in the path have correct dimension
        if k_list.shape[1]!=self._dim_k:
            print('input k-space dimension is',k_list.shape[1])
            print('k-space dimension taken from model is',self._dim_k)
            raise Exception("\n\nk-space dimensions do not match")

        # must have more k-points in the path than number of nodes
        if nk<k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes=k_list.shape[0]
    
        # extract the lattice vectors from the TB model
        lat_per=np.copy(self.model._lat)
        # choose only those that correspond to periodic directions
        lat_per=lat_per[self._per]    
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per,lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node=np.zeros(n_nodes,dtype=float)
        for n in range(1,n_nodes):
            dk = k_list[n]-k_list[n-1]
            dklen = np.sqrt(np.dot(dk,np.dot(k_metric,dk)))
            k_node[n]=k_node[n-1]+dklen
    
        # Find indices of nodes in interpolated list
        node_index=[0]
        for n in range(1,n_nodes-1):
            frac=k_node[n]/k_node[-1]
            node_index.append(int(round(frac*(nk-1))))
        node_index.append(nk-1)
    
        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist=np.zeros(nk,dtype=float)
        #   array listing the interpolated k-points    
        k_vec=np.zeros((nk,self._dim_k),dtype=float)
    
        # go over all kpoints
        k_vec[0]=k_list[0]
        for n in range(1,n_nodes):
            n_i=node_index[n-1]
            n_f=node_index[n]
            kd_i=k_node[n-1]
            kd_f=k_node[n]
            k_i=k_list[n-1]
            k_f=k_list[n]
            for j in range(n_i,n_f+1):
                frac=float(j-n_i)/float(n_f-n_i)
                k_dist[j]=kd_i+frac*(kd_f-kd_i)
                k_vec[j]=k_i+frac*(k_f-k_i)
        return (k_vec,k_dist,k_node)

    def green_surface_cut(self,n_k,E_cut,eta=0.05,N_cut=10,spin_distinguish=False):
        if type(E_cut)!=float:
            raise Exception("Wrong! the E_cut need to be a float, not tuple or array or list")
        if type(n_k)!=int:
            raise Exception("Wrong Input, the n_k need to be a value, not tuple or array or list")
        solve=functools.partial(self.green_solve_one,energy=E_cut,eta=eta,N_cut=N_cut,spin_distinguish=spin_distinguish)
        if self._dim_k==2:
            k=[]
            if spin_distinguish:
                G=np.zeros((n_k,n_k,4),dtype=float)
            else:
                G=np.zeros((n_k,n_k),dtype=float)
            for i in range(n_k):
                for j in range(n_k):
                    k.append([float(i)/float(n_k-1),float(j)/float(n_k-1)])
            pool=Pool()
            G=pool.map(solve,k)
            pool.close()
            pool.join()
            if spin_distinguish:
                G=np.reshape(np.array(G),(n_k,n_k,4))
            else:
                G=np.reshape(np.array(G),(n_k,n_k))
        elif self._dim_k==1:
            raise Exception("Wrong, this only can calculate 3-dimension")
            if spin_distinguish:
                G=np.zeros((n_k,4),dtype=float)
            else:
                G=np.zeros(n_k,dtype=float)
            k=np.zeros(n_k,dtype=float)
            k=np.linspace(0,1,n_k)
            pool=Pool()
            G=pool.map(solve,k)
            pool.close()
            pool.join()
        return G
    def green_solve_one_parallel(self,inputs,eta=0.05,N_cut=10):
        k=inputs[0]
        e=inputs[1]
        G=self.green_solve_one(k,e,eta=eta,N_cut=N_cut)
        return G

    def green_surface_path(self,k_vec,E_window,N_e,eta=0.05,N_cut=10,use_one=True):
        if k_vec.shape[1]!=self._dim_k:
            raise Exception("you must input a path with right arrow")
        if len(E_window)!=2:
            raise Exception("the E_window need to 1*2 list")
        ve=np.linspace(E_window[0],E_window[1],N_e)
        n_k=len(k_vec)
        inputs=[]
        for i in k_vec:
            inputs.append([i,ve])
        if use_one:
            solve=functools.partial(self.green_solve_one_parallel,eta=eta,N_cut=N_cut)
            pool=Pool()
            result=pool.map(solve,inputs)
            pool.close()
            pool.join()
            G=np.reshape(np.array(result),(N_e,n_k,3)).transpose(2,1,0)
        else:
            solve=functools.partial(self.green_solve_one,energy=ve,eta=eta,N_cut=N_cut)
            pool=Pool()
            result=pool.map(solve,k_vec)
            pool.close()
            pool.join()
            G=np.array(result).transpose(2,1,0)
        G=G[:,::-1]
        for i in range(3):
            G[i]-=np.min(G[i])
        G+=1
        return G


class impur(object):
    #这个是用来处理无序的, 主要是用来提取杂志哈密顿量的, 后面还会有 coherent potential approximation
    def __init__(self,model,model_imp,dc,r0,impur):
        r"""
        model 是主要的, 没有杂质的model
        model_imp 是含有一个杂质的模型, 用来提取杂质哈密顿量
        dc 是用来提取哈密顿量时候的一个距离截断
        r0 是指数化衰减时候的一个参数
        impur: 杂质的位置, 用 model_imp 中的原子顺序来指定
        """
        if model._dim_k!=model_imp._dim_k:
            raise Exception("two models must have same k dimension!")
        elif model._dim_r!=model_imp._dim_r:
            raise Exception("two models must have same r dimension!")
        elif model._norb!=model_imp._norb:
            raise Exception("two models must have same number of orbitals")
        elif model._nspin!=model_imp._nspin:
            raise Exception("two models must have same spin")
        elif model._natom!=model_imp._natom:
            raise Exception("two models must have same atoms")
        atom_1=shift_to_zero(np.dot(model._atom_position,model._lat)) #实空间的原子位置
        atom_2=shift_to_zero(np.dot(model_imp._atom_position,model_imp._lat))
        index=[]
        for i in range(model._natom):
            index.append(np.argmin(np.linalg.norm(atom_1[i]-atom_2,axis=1))) 
        # 这里的index[i] 是第一个模型第 i 个原子在第二个模型中对应的原子顺序, 所以 model._orb=model_imp._orb[index]
        index=np.array(index,dtype=int) #如果两个model的原子顺序不一致, 将其调整到一致
        orb_list=np.zeros(model_imp._natom,dtype=int) #model 中, 原子的第一个态在orb 中的位置
        a=0
        for i in range(1,model_imp._natom):
            orb_list[i]=orb_list[i-1]+model_imp._atom[i]
        orb_index=[]
        for i in range(model._natom):
            if model._atom[i]!=model_imp._atom[index[i]]:
                raise Exception("Wrong, the two model have different atoms at one site, please check it, or may be the basis array have different direction, please check again")
            for j in range(model._atom[i]):
                orb_index.append(orb_list[index[i]]+j)
        orb_index=np.array(orb_index,dtype=int)
        if model_imp._nspin==2:
            state_index=np.append(orb_index,orb_index+model._norb)  
        else:
            state_index=orb_index
        ham_1=np.copy(model_imp._ham)[:,state_index][:,:,state_index]#将哈密顿量的原子顺序调整到一致 
        ham_2=np.copy(model._ham) #我们将杂质哈密顿量作为我们的基础变量
        hamR_1=np.copy(model_imp._hamR)
        hamR_2=np.copy(model._hamR)
        impur_ham=np.copy(ham_1)
        impur_hamR=np.copy(hamR_1)
        lat=np.copy(model._lat)
        orb=np.dot(model._orb,lat)
        for i,R in enumerate(hamR_2):
            if np.any(np.all(hamR_1==R,axis=1)):
                index=np.argwhere(np.all(hamR_1==R,axis=1))[0][0]
                impur_ham[index]-=ham_2[i]
            elif np.any(np.all(hamR_1==-R,axis=1)):
                index=np.argwhere(np.all(hamR_1==-R,axis=1))[0][0]
                impur_ham[index]-=ham_2[i].conjugate().T
            else:
                impur_ham=np.append(impur_ham,[-ham_2[i]],axis=0)
                impur_hamR=np.append(impur_hamR,[R],axis=0)
        impur_sites=np.dot(model._atom_position[impur],model._lat) #这里的杂志位置是以纯的原胞的位置为准
        def distance(ind_R,orb,lat,impur_sites): #定义一个计算距离的函数
            dis=np.zeros((len(orb),len(orb)),dtype=float)
            for i in range(len(orb)):
                for j in range(len(orb)):
                    dis[i,j]=np.linalg.norm(orb[i]-impur_sites)+np.linalg.norm(orb[j]-impur_sites)
            R0=np.linalg.norm(np.dot(ind_R,lat)) 
            dis+=R0
            return dis
        index=np.logical_not((impur_ham==0).all(axis=2).all(axis=1))
        impur_ham=impur_ham[index]
        impur_hamR=impur_hamR[index]
        for i,R in enumerate(impur_hamR):
            dis=distance(R,orb,lat,impur_sites)
            if model_imp._nspin==2:
                dis=np.kron([[1,1],[1,1]],dis)
            phase=np.exp(-(dis/r0)**8)
            impur_ham[i]*=(dis<dc)*phase
        index=np.logical_not((impur_ham==0).all(axis=2).all(axis=1))
        self._ham=impur_ham[index]
        self._hamR=impur_hamR[index]
        self._lat=np.copy(model._lat)
        self._orb=np.copy(model._orb)
        self._atom_position=np.copy(model._atom_position)
        self._atom=np.copy(model._atom)
        self._natom=len(model._atom)
        self._impur_site=impur
        self._norb=len(self._orb)
        self._nspin=model._nspin
        self._per=model._per
        self._dim_r=model._dim_r
        self._dim_k=model._dim_k
        if self._nspin==2:
            self._nsta=self._norb*2
        else:
            self._nsta=self._norb


    def impur_shift(self,R,judge=1e-5): # 这个是用来进行平移的
        new_atom_position=shift_to_zero(self._atom_position+R)%1
        if same_atom(self._atom_position,new_atom_position,judge)==False:
            raise Exception("Wrong, the R is not in the space group")
        atom_match=match_atom(self._atom_position,new_atom_position)
        inv_atom_match=match_atom(new_atom_position,self._atom_position)
        new_atom=self._atom[atom_match]
        new_hamR=np.copy(self._hamR)
        new_ham=np.copy(self._ham)
        new_natom=self._natom
        orb_match=np.zeros(self._norb,dtype=int)
        inv_orb_match=np.zeros(self._norb,dtype=int)
        new_orb_list=np.zeros(self._natom,dtype=int)
        new_orb_list=gen_orb_list(new_orb_list,self._natom,self._atom)
        old_orb_list=np.zeros(self._natom,dtype=int)
        old_orb_list=gen_orb_list(old_orb_list,self._natom,self._atom)
        orb_match=gen_orb_match(orb_match,old_orb_list,atom_match,self._atom)
        inv_orb_match=gen_orb_match(inv_orb_match,new_orb_list,inv_atom_match,self._atom)
        shift_R=np.array(shift_to_zero((self._atom_position+R)-new_atom_position),dtype=int)
        shift_orb_R=np.zeros((self._norb,self._dim_r),dtype=int)
        a=0
        for i in range(self._natom):
            for j in range(new_atom[i]):
                shift_orb_R[a]=shift_R[i]
                a+=1
        new_orb=self._orb[orb_match]
        new_hamR=np.zeros((1,self._dim_r),dtype=int)
        new_ham=np.zeros((1,self._nsta,self._nsta),dtype=complex)
        for R,hamR in enumerate(self._hamR):
            for i in range(self._norb):
                new_orb_i=orb_match[i]
                for j in range(self._norb):
                    new_orb_j=orb_match[j]
                    use_hamR=hamR-shift_orb_R[i]+shift_orb_R[j]
                    index_R=(new_hamR==use_hamR).all(axis=1)
                    inv_index_R=(new_hamR==-use_hamR).all(axis=1)
                    if self._nspin==1:
                        hop=self._ham[R,i,j]
                        if index_R.any():
                            index=np.argwhere(index_R)[0][0]
                            new_ham[index][new_orb_i,new_orb_j]=hop
                        elif inv_index_R.any():
                            index=np.argwhere(inv_index_R)[0][0]
                            new_ham[index][new_orb_j,new_orb_i]=hop.conjugate()
                        else:
                            use_ham=np.zeros((self._nsta,self._nsta),dtype=complex)
                            use_ham[new_orb_i,new_orb_j]=hop
                            new_ham=np.append(new_ham,[use_ham],axis=0)
                            new_hamR=np.append(new_hamR,[use_hamR],axis=0)
                    else:
                        hop=self._ham[R,i::self._norb,j::self._norb]
                        if index_R.any():
                            index=np.argwhere(index_R)[0][0]
                            new_ham[index][new_orb_i::self._norb,new_orb_j::self._norb]=hop
                        elif inv_index_R.any(): 
                            index=np.argwhere(inv_index_R)[0][0]
                            new_ham[index][new_orb_j::self._norb,new_orb_i::self._norb]=hop.conjugate().T
                        else:
                            use_ham=np.zeros((self._nsta,self._nsta),dtype=complex)
                            use_ham[new_orb_i::self._norb,new_orb_j::self._norb]=hop
                            new_ham=np.append(new_ham,[use_ham],axis=0)
                            new_hamR=np.append(new_hamR,[use_hamR],axis=0)
        self._impur_site=inv_atom_match[self._impur_site]
        self._ham=new_ham
        self._hamR=new_hamR




    def plus_impur(self,model_old):
        model=copy.deepcopy(model_old)
        if self._hamR.shape[1]!=model._hamR.shape[1]:
            raise Exception("Wrong, the dimension of impur hamiltonian must equal to the crystal")
        if self._ham.shape[1]!=model._ham.shape[1]:
            raise Exception("Wrong, the dimension of impur hamiltonian must equal to the crystal")
        if self._norb!=model._norb:
            raise Exception("Wrong, two model must have same orbit number")
        ham=np.copy(self._ham)
        hamR=np.copy(self._hamR)
        #############先将两个模型的原子对齐###################################
        atom_1=np.dot(self._atom_position,self._lat)
        atom_2=np.dot(model._atom_position,model._lat) #实空间的原子位置
        index=[]
        for i in range(model._natom):
            index.append(np.argmin(np.linalg.norm(atom_1-atom_2[i],axis=1))) 
        # 这里的index[i] 是第二个模型第 i 个原子在第一个模型中对应的原子顺序, 所以 self._orb[index]=model._orb
        index=np.array(index,dtype=int) #如果两个model的原子顺序不一致, 将其调整到一致
        orb_list=np.zeros(model._natom,dtype=int) #model 中, 原子的第一个态在orb 中的位置
        a=0
        for i in range(self._natom):
            orb_list[i]=a
            for j in range(self._atom[i]):
                a+=1
        orb_index=[]
        for i in range(model._natom):
            if model._atom[i]!=self._atom[index[i]]:
                raise Exception("Wrong, the two model have different atoms at one site, please check it, or may be the basis array have different direction, please check again")
            for j in range(model._atom[i]):
                orb_index.append(orb_list[index[i]]+j)
        orb_index=np.array(orb_index,dtype=int)
        if self._nspin==2:
            state_index=np.append(orb_index,orb_index+model._norb) #如果有自旋, 将顺序乘以 2
        else:
            state_index=orb_index
        ham=ham[:,state_index][:,:,state_index]
        self._impur_site=index[self._impur_site]
        ####################################################################################
        for i,R in enumerate(hamR):
            if np.any(np.all(model._hamR==R,axis=1)):
                index=np.argwhere(np.all(model._hamR==R,axis=1))[0][0]
                model._ham[index]+=ham[i]
            elif np.any(np.all(model._hamR==-R,axis=1)):
                index=np.argwhere(np.all(model._hamR==-R,axis=1))[0][0]
                model._ham[index]+=ham[i].conjugate().T
            else:
                model._ham=np.append(model._ham,[ham[i]],axis=0)
                model._hamR=np.append(model._hamR,[R],axis=0)
        return model

    def visualize(self,dir_first,dir_second=None,draw_hoppings=False,eig_dr=None,ph_color="black"):
        r"""

        Rudimentary function for visualizing tight-binding model geometry,
        hopping between tight-binding orbitals, and electron eigenstates.

        If eigenvector is not drawn, then orbitals in home cell are drawn
        as red circles, and those in neighboring cells are drawn with
        different shade of red. Hopping term directions are drawn with
        green lines connecting two orbitals. Origin of unit cell is
        indicated with blue dot, while real space unit vectors are drawn
        with blue lines.

        If eigenvector is drawn, then electron eigenstate on each orbital
        is drawn with a circle whose size is proportional to wavefunction
        amplitude while its color depends on the phase. There are various
        coloring schemes for the phase factor; see more details under
        *ph_color* parameter. If eigenvector is drawn and coloring scheme
        is "red-blue" or "wheel", all other elements of the picture are
        drawn in gray or black.

        :param dir_first: First index of Cartesian coordinates used for
          plotting.

        :param dir_second: Second index of Cartesian coordinates used for
          plotting. For example if dir_first=0 and dir_second=2, and
          Cartesian coordinates of some orbital is [2.0,4.0,6.0] then it
          will be drawn at coordinate [2.0,6.0]. If dimensionality of real
          space (*dim_r*) is zero or one then dir_second should not be
          specified.

        :param eig_dr: Optional parameter specifying eigenstate to
          plot. If specified, this should be one-dimensional array of
          complex numbers specifying wavefunction at each orbital in
          the tight-binding basis. If not specified, eigenstate is not
          drawn.

        :param ph_color: Optional parameter determining the way
          eigenvector phase factors are translated into color. Default
          value is "black". Convention of the wavefunction phase is as
          in convention 1 in section 3.1 of :download:`notes on
          tight-binding formalism  <misc/pythtb-formalism.pdf>`.  In
          other words, these wavefunction phases are in correspondence
          with cell-periodic functions :math:`u_{n {\bf k}} ({\bf r})`
          not :math:`\Psi_{n {\bf k}} ({\bf r})`.

          * "black" -- phase of eigenvectors are ignored and wavefunction
            is always colored in black.

          * "red-blue" -- zero phase is drawn red, while phases or pi or
            -pi are drawn blue. Phases in between are interpolated between
            red and blue. Some phase information is lost in this coloring
            becase phase of +phi and -phi have same color.

          * "wheel" -- each phase is given unique color. In steps of pi/3
            starting from 0, colors are assigned (in increasing hue) as:
            red, yellow, green, cyan, blue, magenta, red.

        :returns:
          * **fig** -- Figure object from matplotlib.pyplot module
            that can be used to save the figure in PDF, EPS or similar
            format, for example using fig.savefig("name.pdf") command.
          * **ax** -- Axes object from matplotlib.pyplot module that can be
            used to tweak the plot, for example by adding a plot title
            ax.set_title("Title goes here").

        Example usage::

          # Draws x-y projection of tight-binding model
          # tweaks figure and saves it as a PDF.
          (fig, ax) = tb.visualize(0, 1)
          ax.set_title("Title goes here")
          fig.savefig("model.pdf")

        See also these examples: :ref:`edge-example`,
        :ref:`visualize-example`.

        """

        # check the format of eig_dr
        if not (eig_dr is None):
            if eig_dr.shape!=(self._nsta,):
                raise Exception("\n\nWrong format of eig_dr! Must be array of size norb.")

        # check that ph_color is correct
        if ph_color not in ["black","red-blue","wheel"]:
            raise Exception("\n\nWrong value of ph_color parameter!")

        # check if dir_second had to be specified
        if dir_second==None and self._dim_r>1:
            raise Exception("\n\nNeed to specify index of second coordinate for projection!")

        # start a new figure
        import pylab as plt
        fig=plt.figure(figsize=[plt.rcParams["figure.figsize"][0],
                                plt.rcParams["figure.figsize"][0]])
        ax=fig.add_subplot(111, aspect='equal')

        def proj(v):
            "Project vector onto drawing plane"
            coord_x=v[dir_first]
            if dir_second==None:
                coord_y=0.0
            else:
                coord_y=v[dir_second]
            return [coord_x,coord_y]

        def to_cart(red):
            "Convert reduced to Cartesian coordinates"
            return np.dot(red,self._lat)

        # define colors to be used in plotting everything
        # except eigenvectors
        if (eig_dr is None) or ph_color=="black":
            c_cell="b"
            c_orb="r"
            c_nei=[0.85,0.65,0.65]
            c_hop="g"
        else:
            c_cell=[0.4,0.4,0.4]
            c_orb=[0.0,0.0,0.0]
            c_nei=[0.6,0.6,0.6]
            c_hop=[0.0,0.0,0.0]
        # determine color scheme for eigenvectors
        def color_to_phase(ph):
            if ph_color=="black":
                return "k"
            if ph_color=="red-blue":
                ph=np.abs(ph/np.pi)
                return [1.0-ph,0.0,ph]
            if ph_color=="wheel":
                if ph<0.0:
                    ph=ph+2.0*np.pi
                ph=6.0*ph/(2.0*np.pi)
                x_ph=1.0-np.abs(ph%2.0-1.0)
                if ph>=0.0 and ph<1.0: ret_col=[1.0 ,x_ph,0.0 ]
                if ph>=1.0 and ph<2.0: ret_col=[x_ph,1.0 ,0.0 ]
                if ph>=2.0 and ph<3.0: ret_col=[0.0 ,1.0 ,x_ph]
                if ph>=3.0 and ph<4.0: ret_col=[0.0 ,x_ph,1.0 ]
                if ph>=4.0 and ph<5.0: ret_col=[x_ph,0.0 ,1.0 ]
                if ph>=5.0 and ph<=6.0: ret_col=[1.0 ,0.0 ,x_ph]
                return ret_col

        # draw origin
        ax.plot([0.0],[0.0],"o",c=c_cell,mec="w",mew=0.0,zorder=7,ms=4.5)

        # first draw unit cell vectors which are considered to be periodic
        for i in self._per:
            # pick a unit cell vector and project it down to the drawing plane
            vec=proj(self._lat[i])
            ax.plot([0.0,vec[0]],[0.0,vec[1]],"-",c=c_cell,lw=1.5,zorder=7)

        # now draw all orbitals
        for i in range(self._norb):
            # find position of orbital in cartesian coordinates
            pos=to_cart(self._orb[i])
            pos=proj(pos)
            ax.plot([pos[0]],[pos[1]],"o",c=c_orb,mec="w",mew=0.0,zorder=10,ms=4.0)

        orb_list=np.zeros(self._natom,dtype=int)
        orb_list=gen_orb_list(orb_list,self._natom,self._atom)
        draw_hamR=np.zeros((1,self._dim_r),dtype=int)
        draw_orb=np.zeros((1,2),dtype=int)
        if draw_hoppings==True: #画hopping
            for r,R in enumerate(self._hamR):
                for i in range(self._norb):
                    for j in range(self._norb):
                        if np.abs(self._ham[r][i,j])!=0:
                            draw_hamR=np.append(draw_hamR,[R],axis=0) #找到非0 hopping
                            draw_orb=np.append(draw_orb,[[i,j]],axis=0)
            for s0,h in enumerate(draw_hamR):
                for s in range(2):
                    i=draw_orb[s0,0]
                    j=draw_orb[s0,1]
                    pos_i=np.copy(self._orb[i])
                    pos_j=np.copy(self._orb[j])
                    if self._dim_k!=0:
                        if s==0:
                            pos_j[self._per]=pos_j[self._per]+h[self._per]
                        if s==1:
                            pos_i[self._per]=pos_i[self._per]-h[self._per]
                    pos_i=np.array(proj(to_cart(pos_i)))
                    pos_j=np.array(proj(to_cart(pos_j)))
                    all_pnts=np.array([pos_i,pos_j]).T
                    ax.plot(all_pnts[0],all_pnts[1],"-",c=c_hop,lw=0.75,zorder=8)
                    ax.plot([pos_i[0]],[pos_i[1]],"o",c=c_nei,zorder=9,mew=0.0,ms=4.0,mec="w")
                    ax.plot([pos_j[0]],[pos_j[1]],"o",c=c_nei,zorder=9,mew=0.0,ms=4.0,mec="w")
        # draw hopping terms

        # now draw the eigenstate
        if not (eig_dr is None):
            if self._nspin==2:
                orb=np.append(self._orb,self._orb,axis=0)
            else:
                orb=self._orb
            for i in range(self._nsta):
                # find position of orbital in cartesian coordinates

                pos=to_cart(orb[i])
                pos=proj(pos)
                # find norm of eigenfunction at this point
                nrm=(eig_dr[i]*eig_dr[i].conjugate()).real
                # rescale and get size of circle
                nrm_rad=2.0*nrm*float(self._norb)
                # get color based on the phase of the eigenstate
                phase=np.angle(eig_dr[i])
                c_ph=color_to_phase(phase)
                ax.plot([pos[0]],[pos[1]],"o",c=c_ph,mec="w",mew=0.0,ms=nrm_rad,zorder=11,alpha=0.8)

        # center the image
        #  first get the current limit, which is probably tight
        xl=ax.set_xlim()
        yl=ax.set_ylim()
        # now get the center of current limit
        centx=(xl[1]+xl[0])*0.5
        centy=(yl[1]+yl[0])*0.5
        # now get the maximal size (lengthwise or heightwise)
        mx=max([xl[1]-xl[0],yl[1]-yl[0]])
        # set new limits
        extr=0.05 # add some boundary as well
        ax.set_xlim(centx-mx*(0.5+extr),centx+mx*(0.5+extr))
        ax.set_ylim(centy-mx*(0.5+extr),centy+mx*(0.5+extr))

        # return a figure and axes to the user
        return (fig,ax)
    
    def unfold_impur(self,U):
        #这个程序是将超胞中提取出来的杂志哈密顿量用原胞进行表示
        U_inv=np.linalg.inv(U)
        U_det=np.linalg.det(U)
        new_lat=np.dot(U_inv,self._lat)
        new_atom_position=np.dot(self._atom_position,U)
        new_orb=np.dot(self._orb,U)%1


def get_rv(orb):
    norb=len(orb)
    dim_r=len(orb[0])
    rv=np.zeros((norb,norb,dim_r),dtype=complex)
    for i in range(norb):
        for j in range(norb):
            rv[i,j]=-orb[i]+orb[j]
    return rv

def plus(model_1,model_2,lm,consider_distance=True,pow_index=1):
    r"""
    this pocisdure is plus model_1 and model_2, with parameterlm, new_model=(1-lm)*model_1+lm*model_2, but have (1-lm^2)*model_1+lm*model_2. two model must have same orbits, same lats, and save r,k dimensions
    r"""

    if model_1._dim_k!=model_2._dim_k:
        raise Exception("two models must have same k dimension!")
    elif model_1._dim_r!=model_2._dim_r:
        raise Exception("two models must have same r dimension!")
    elif model_1._norb!=model_2._norb:
        raise Exception("two models must have same number of orbitals")
    elif model_1._nspin!=model_2._nspin:
        raise Exception("two models must have same spin")
    elif lm<0 or lm>1:
        raise Exception("lm must setisfy 0<=lm<=1")
    dim_k=model_1._dim_k
    dim_r=model_1._dim_r
    natom=model_1._natom
    norb=model_1._norb
    #########generate new lat and orbi###############
    new_lat=(1-lm)*model_1._lat+lm*model_2._lat
    c_lat_1=np.dot(model_1._orb,model_1._lat) #model_1's orbit coordinate in Cartesian
    c_lat_2=np.dot(model_2._orb,model_2._lat)
    c_atom_1=np.dot(model_1._atom_position,model_1._lat)
    c_atom_2=np.dot(model_2._atom_position,model_2._lat)
    index=match_atom(c_atom_1,c_atom_2)
    index=np.array(index,dtype=int)
    c_atom_2=c_atom_2[index]
    c_lat_new=(1-lm)*c_lat_1+lm*c_lat_2
    new_atom_position=(1-lm)*c_atom_1+lm*c_atom_2
    new_atom=np.copy(model_1._atom)
    new_orb=np.dot(c_lat_new,np.linalg.inv(new_lat))
    new_atom_position=np.dot(new_atom_position,np.linalg.inv(new_lat))
    new_model=tb_model(dim_k,dim_r,new_lat,new_orb,atom_list=new_atom,atom_position=new_atom_position,nspin=model_1._nspin)
    ham_1=model_1._ham*(1-lm**pow_index)
    ham_2=model_2._ham*lm
    hamR_1=model_1._hamR
    hamR_2=model_2._hamR
    def gen_new_ham(orb,orb_new,norb,hamR,ham,nspin):
        for r,R in enumerate(hamR):
            for i in range(norb):
                for j in range(norb):
                    distance=np.linalg.norm(orb[j]-orb[i]+hamR)
                    distance_new=np.linalg.norm(orb_new[j]-orb_new[i]+hamR)
                    if distance_new!=0:
                        ham[r,i,j]*=(distance/distance_new)**2
                    if nspin==2:
                        ham[r,i+norb,j]*=(distance/distance_new)**2
                        ham[r,i,j+norb]*=(distance/distance_new)**2
                        ham[r,i+norb,j+norb]*=(distance/distance_new)**2
        return ham
    if consider_distance:
        ham_1=gen_new_ham(c_lat_1,c_lat_new,norb,hamR_1,ham_1,model_1._nspin)
        ham_2=gen_new_ham(c_lat_2,c_lat_new,norb,hamR_2,ham_2,model_2._nspin)
    new_model._ham=ham_1
    new_model._hamR=hamR_1
    for i,R in enumerate(hamR_2):
        if np.any(np.all(new_model._hamR==R,axis=1)):
            index=np.argwhere(np.all(new_model._hamR==R,axis=1))[[0]]
            new_model._ham[index]+=ham_2[i]
        elif np.any(np.all(new_model._hamR==-R,axis=1)):
            index=np.argwhere(np.all(new_model._hamR==-R,axis=1))[[0]]
            new_model._ham[index]+=ham_2[i].conjugate().T
        else:
            new_model._hamR=np.append(new_model._hamR,[R],axis=0)
            new_model._ham=np.append(new_model._ham,[ham_2[i]],axis=0)
    return new_model


def _one_berry_loop(wf,berry_evals=False):
    """Do one Berry phase calculation (also returns a product of M
    matrices).  Always returns numbers between -pi and pi.  wf has
    format [kpnt,band,orbital,spin] and kpnt has to be one dimensional.
    Assumes that first and last k-point are the same. Therefore if
    there are n wavefunctions in total, will calculate phase along n-1
    links only!  If berry_evals is True then will compute phases for
    individual states, these corresponds to 1d hybrid Wannier
    function centers. Otherwise just return one number, Berry phase."""
    # number of occupied states
    nocc=wf.shape[1]
    # temporary matrices
    prd=np.identity(nocc,dtype=complex)
    ovr=np.zeros([nocc,nocc],dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    if len(wf.shape)==4:
      wf=np.reshape(wf,(wf.shape[0],wf.shape[1],wf.shape[2]*wf.shape[3]))
      wf1=wf[1:]
      wf=wf[0:-1]
      ovr=np.einsum('ijm,ikm->ijk',wf.conjugate(),wf1,optimize=True)
    elif len(wf.shape)==3:
      wf1=wf[1:]
      wf=wf[0:-1]
      ovr=np.einsum('ijm,ikm->ijk',wf.conjugate(),wf1,optimize=True)
    elif len(wf.shape)==2:
      wf1=wf[1:]
      wf=wf[0:-1]
      ovr=np.einsum('im,im->i',wf.conjugate(),wf1,optimize=True)
      pha=(-1.0)*np.angle(ovr)
      return pha
    if berry_evals==False:
        for i in range(wf.shape[0]):
            prd=np.dot(prd,ovr[i])
    else:
        for i in range(wf.shape[0]):
            matU,sing,matV=np.linalg.svd(ovr[i])
            prd=np.einsum('ij,jk,kl->il',prd,matU,matV,optimize=True)
    # calculate Berry phase
    if berry_evals==False:
        det=np.linalg.det(prd)
        pha=(-1.0)*np.angle(det)
        return pha
    # calculate phases of all eigenvalues
    else:
        evals=np.linalg.eigvals(prd)
        eval_pha=(-1.0)*np.angle(evals)
        # sort these numbers as well
        eval_pha=np.sort(eval_pha)
        return eval_pha

def __wilson_loop__(wf,berry_evals=False):
    nocc=wf.shape[1]
    # temporary matrices
    prd=np.identity(nocc,dtype=complex)
    ovr=np.zeros([nocc,nocc],dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    wf1=wf[1:]
    for i in range(wf.shape[0]-1):
        # generate overlap matrix, go over all bands
        #for j in range(nocc):
        #    for k in range(nocc):
        #        ovr[j,k]=_wf_dpr(wf[i,j,:],wf[i+1,k,:])
        #ovr=np.einsum('jm,km->jk',wf[i].conjugate(),wf1[i],optimize=True)
        ovr=np.dot(wf[i].conjugate(),wf1[i].T)
        # only find Berry phase
        if berry_evals==False:
            # multiply overlap matrices
            prd=np.dot(prd,ovr)
        # also find phases of individual eigenvalues
        else:
            # cleanup matrices with SVD then take product
            matU,sing,matV=np.linalg.svd(ovr)
            prd=np.dot(prd,np.dot(matU,matV))
    return prd
    
def _one_flux_plane(wfs2d):
    "Compute fluxes on a two-dimensional plane of states."
    # size of the mesh
    nk0=wfs2d.shape[0]
    nk1=wfs2d.shape[1]
    # number of bands (will compute flux of all bands taken together)
    nbnd=wfs2d.shape[2]
    norb=wfs2d.shape[3]

    # here store flux through each plaquette of the mesh
    all_phases=np.zeros((nk0-1,nk1-1),dtype=float)

    # go over all plaquettes
    wf_use=np.zeros(((nk0-1)*(nk1-1),5,nbnd,norb),dtype=complex)
    for i in range(nk0-1):
        for j in range(nk1-1):
            # generate a small loop made out of four pieces
            wf_use[i*(nk1-1)+j,0]=wfs2d[i,j]
            wf_use[i*(nk1-1)+j,1]=wfs2d[i+1,j]
            wf_use[i*(nk1-1)+j,2]=wfs2d[i+1,j+1]
            wf_use[i*(nk1-1)+j,3]=wfs2d[i,j+1]
            wf_use[i*(nk1-1)+j,4]=wfs2d[i,j]
    # calculate phase around one plaquette
    pool=Pool()
    results=pool.map(_one_berry_loop,wf_use)
    pool.close()
    pool.join()
    results=np.array(results)
    all_phases=np.reshape(results,(nk0-1,nk1-1,nbnd))
    return all_phases

def no_2pi(x,clos):
    "Make x as close to clos by adding or removing 2pi"
    while abs(clos-x)>np.pi:
        if clos-x>np.pi:
            x+=2.0*np.pi
        elif clos-x<-1.0*np.pi:
            x-=2.0*np.pi
    return x

def _one_phase_cont(pha,clos):
    """Reads in 1d array of numbers *pha* and makes sure that they are
    continuous, i.e., that there are no jumps of 2pi. First number is
    made as close to *clos* as possible."""
    ret=np.copy(pha)
    # go through entire list and "iron out" 2pi jumps
    for i in range(len(ret)):
        # which number to compare to
        if i==0: cmpr=clos
        else: cmpr=ret[i-1]
        # make sure there are no 2pi jumps
        ret[i]=no_2pi(ret[i],cmpr)
    return ret

def _array_phases_cont(arr_pha,clos):
    """Reads in 2d array of phases *arr_pha* and makes sure that they
    are continuous along first index, i.e., that there are no jumps of
    2pi. First array of phasese is made as close to *clos* as
    possible."""
    ret=np.zeros_like(arr_pha)
    # go over all points
    for i in range(arr_pha.shape[0]):
        # which phases to compare to
        if i==0: cmpr=clos
        else: cmpr=ret[i-1,:]
        # remember which indices are still available to be matched
        avail=list(range(arr_pha.shape[1]))
        # go over all phases in cmpr[:]
        for j in range(cmpr.shape[0]):
            # minimal distance between pairs
            min_dist=1.0E10
            # closest index
            best_k=None
            # go over each phase in arr_pha[i,:]
            for k in avail:
                cur_dist=np.abs(np.exp(1.0j*cmpr[j])-np.exp(1.0j*arr_pha[i,k]))
                if cur_dist<=min_dist:
                    min_dist=cur_dist
                    best_k=k
            # remove this index from being possible pair later
            avail.pop(avail.index(best_k))
            # store phase in correct place
            ret[i,j]=arr_pha[i,best_k]
            # make sure there are no 2pi jumps
            ret[i,j]=no_2pi(ret[i,j],cmpr[j])
    return ret
def gcd(a):
    a=np.array(a)
    if (a==np.zeros_like(a)).all():
        raise Exception("\n\n Error, the vec elements can't all be zero!")
    b=np.gcd.reduce(a)
    return a/b
#    b=a.min()
#    if b!=0:
#      if b!=1:
#        for i in range(1,b):
#          if (np.mod(a,b-i)==np.zeros_like(a)).all():
#            return a/(b-i)
#      else:
#        return a
#    else:
#      b=np.delete(a,a==0)
#      b=gcd(b)
#      a[a!=0]=b
#      return a

class nest_wf_array(object):
    r"""
    This class is used to calculate the nested Wilson loop.

    """
    def __init__(self,wf,occ,dir=0):
        self.wf_array=copy.deepcopy(wf)
        self.model=copy.deepcopy(wf._model)
        self._nspin=wf._nspin
        occ=np.array(occ,dtype=int)
        # number of k_dim
        self._dim_arr=wf._dim_arr-1
        # number of orbitals
        if wf._nspin==1:
            self._orb=wf._model._orb[occ]
        self._norb=wf._model._norb
        self._nsta=self._norb*self._nspin
        self._mesh_arr=np.array(wf._mesh_arr,dtype=int)
        if True in (self._mesh_arr<=1).tolist():
            raise Exception("\n\nDimension of wf_array object in each direction must be 2 or larger.")
        #self._mesh_arr=np.array(wf._mesh_arr[1:2])
        self._mesh_arr=np.delete(self._mesh_arr,dir)
        wfs_dim=np.copy(self._mesh_arr)
        wfs_dim=np.append(wfs_dim,len(occ))
        self._evl=np.zeros(wfs_dim,dtype=complex)
        wfs_dim=np.append(wfs_dim,self._nsta)
        self._wfs=np.zeros(wfs_dim,dtype=complex)
        self._per=range(self._dim_arr)
        self._dir=dir
        self._occ=occ
        loop=wf.Wilson_loop(self._occ,self._dir,berry_evals=True)
        self._loop=loop
          
    def solve_on_grid(self):
        if self._norb<=1:
            all_gaps=None
        else:
            gap_dim=np.copy(self._mesh_arr)
            gap_dim=np.append(gap_dim,self._norb-1)
            all_gaps=np.zeros(gap_dim,dtype=float)
        if self._dim_arr==1:
            for i in range(self._mesh_arr[0]):
                kpt=[i]
                (eval,evec)=self.solve_one(kpt)
                if self._dir==0:
                    evecs=self.wf_array._wfs[0,i,self._occ,...]
                    self._wfs[i]=np.dot(evec,evecs)
                else:
                    evecs=self.wf_array._wfs[i,0,self._occ,...]
                    self._wfs[i]=np.dot(evec,evecs)
                if all_gaps is not None:
                    all_gaps[i,:]=eval[1:]-eval[:-1]
        #self.impose_pbc(0,self._per[0])
        elif self._dim_arr==2:
            for i in range(self._mesh_arr[0]):
                for j in range(self._mesh_arr[1]):
                    #kpt=[start_k[0]+float(i)/float(self._mesh_arr[0]-1),\
                    #    start_k[1]+float(i)/float(self._mesh_arr[1]-1)]
                    kpt=[i,j]
                    (eval,evec)=self.solve_one(kpt)
                    if self._dir==0:
                        evecs=self.wf_array._wfs[0,i,j,self._occ,...]
                        self._wfs[i,j]=np.dot(evec,evecs)
                    elif self._dir==1:
                        evecs=self.wf_array._wfs[i,0,j,self._occ,...]
                        self._wfs[i,j]=np.dot(evec,evecs)
                    elif self._dir==2:
                        evecs=self.wf_array._wfs[i,j,0,self._occ,...]
                        self._wfs[i,j]=np.dot(evec,evecs)
                    else:
                        raise Exception("\n\nDimension of nested_wf_array must be 2 or less.")
                    if all_gaps is not None:
                        all_gaps[i,j,:]=eval[1:]-eval[:-1]
            #for dir in range(2):
              #self.impose_pbc(dir,self._per[dir])
        else:
            raise Exception("\n\nWrong dimensionality!")
        if all_gaps is not None:
            return all_gaps.min(axis=tuple(range(self._dim_arr)))

    def solve_one(self,kpt,eig_vectors=True):
        if len(kpt)==1:
            loop=self._loop[kpt[0],...]
        elif len(kpt)==2:
            loop=self._loop[kpt[0],kpt[1],...]
        else:
            raise Exception("\n\n Wrong k_point dimension")
        if eig_vectors==False:
            eval=-np.angle(np.linalg.eigvalsh(loop))
            eval=eval%(2*np.pi)
            eval=_nicefy_eig(eval)
        else:
            (eval,eig)=np.linalg.eig(loop)
            eval=-np.angle(eval)
            eval=eval%(2*np.pi)
            eig=eig.T
            (eval,eig)=_nicefy_eig(eval,eig)
            return (eval,eig)

    def berry_phase(self,occ,dir=None,berry_evals=False,contin=True):
        if self.model._assume_position_operator_diagonal==False:
            _offdiag_approximation_warning_and_stop()
        if self._dim_arr==1:
            ret=[]
            wf_use=self._wfs[:,occ,:]
            ret=_nested_berry_loop(wf_use,berry_evals)
            return ret
        elif self._dim_arr==2:
            if dir==0:
                ret=[]
                for i in range(self._mesh_arr[1]):
                    wf_use=self._wfs[:,i,...][:,occ,:]
                    ret.append(_nested_berry_loop(wf_use,berry_evals))
            elif dir==1:
                ret=[]
                for i in range(self._mesh_arr[0]):
                    wf_use=self._wfs[i,:,...][:,occ,:]
                    ret.append(_nested_berry_loop(wf_use,berry_evals))
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        else:
            raise Exception("\n\nWrong dimensionality!")
        if self._dim_arr>1 or berry_evals==True:
            ret=np.array(ret,dtype=float)

        if contin==True:
            if berry_evals==False:
                if self._dim_arr==2:
                    ret=_one_phase_cont(ret,ret[0])
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
            else:
                if self._dim_arr==2:
                    ret=_array_phases_cont(ret,ret[0,:])
                elif self._dim_arr!=1:
                    raise Exception("\n\nWrong dimensionality!")
        ret.sort()
        return ret

def _nested_berry_loop(wf,berry_evals):
   # number of occupied states
    nocc=wf.shape[1]
    # temporary matrices
    prd=np.identity(nocc,dtype=complex)
    ovr=np.zeros([nocc,nocc],dtype=complex)
    # go over all pairs of k-points, assuming that last point is overcounted!
    wf1=wf[1:]
    for i in range(wf.shape[0]-1):
        # generate overlap matrix, go over all bands
        ovr=np.dot(wf[i].conjugate(),wf1[i].T)
        if berry_evals==False:
            # multiply overlap matrices
            prd=np.dot(prd,ovr)
        # also find phases of individual eigenvalues
        else:
            # cleanup matrices with SVD then take product
            matU,sing,matV=np.linalg.svd(ovr)
            prd=np.dot(prd,np.dot(matU,matV))
    # calculate Berry phase
    if berry_evals==False:
        det=np.linalg.det(prd)
        pha=(1.0)*np.angle(det)
        return pha
    # calculate phases of all eigenvalues
    else:
        evals=np.linalg.eigvals(prd)
        eval_pha=(1.0)*np.angle(evals)
        # sort these numbers as well
        eval_pha=np.sort(eval_pha)
        return eval_pha

@njit
def gen_w90_ham(n_R,n_orb,tmp,weight,ham):
    for r in range(n_R):
        for i in range(n_orb):
            for j in range(n_orb):
                ham[r,j,i]=tmp[r*n_orb**2+i*n_orb+j]/weight[r]
    return ham
@njit
def gen_w90_r(n_R,n_orb,tmp,rmatrix):
    for r in range(n_R):
        for i in range(n_orb):
            for j in range(n_orb):
                for s in range(3):
                    rmatrix[s,r,j,i]=tmp[r*n_orb**2+i*n_orb+j,s]
    return rmatrix

def k_mesh(n_k,n_dim,mode="full"):
    if mode=="full":
        x=np.linspace(0,1,n_k)
    elif mode=="half":
        x=np.linspace(0,0.5,n_k)
    k=np.zeros((n_k**n_dim,n_dim),dtype=float)
    if n_dim==3:
        for i in range(n_k):
            for j in range(n_k):
                for n in range(n_k):
                    k[i*n_k**2+j*n_k+n]=x[[i,j,n]]
    elif n_dim==2:
        for i in range(n_k):
            for j in range(n_k):
                k[i*n_k+j]=x[[i,j]]
    elif n_dim==1:
        k=x
    return k
 
def gauss_smair(x,y,sigma):
    if len(y)!=len(x):
        raise Exception("the data x and y must have same dimension")
    x0=(x[-1]+x[0])/2
    y0=1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2))
    return np.convolve(y,y0,mode='same')

def gauss_function(x,sigma):
    return np.exp(-x**2/sigma**2/2)/np.sqrt(2*np.pi)/sigma

def _nicefy_eig(eval,eig=None):
    "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
    # first take only real parts of the eigenvalues
    eval=np.array(eval.real,dtype=float)
    # sort energies
    args=eval.argsort()
    eval=eval[args]
    if not (eig is None):
        eig=eig[args]
        return (eval,eig)
    return eval

def shift_to_zero(a):
    a=np.array(a)
    if np.any(np.abs(a-np.floor(a))<1e-5):
        index=np.abs(a-np.floor(a))<1e-5
        a[index]=np.round(a[index])
    if np.any(np.abs(a-np.ceil(a))<1e-5):
        index=np.abs(a-np.ceil(a))<1e-5
        a[index]=np.round(a[index])
    return a

def match_atom(atom_1,atom_2):
    #返回第二个原子相对于第一个原子的位置
    natom=len(atom_1)
    if len(atom_1)!=len(atom_2):
        raise Exception("Wrong!, two atom list must be same number")
    index=[]
    for i in range(natom):
        index.append(np.argmin(np.linalg.norm(atom_1-atom_2[i],axis=1)))
    index=np.array(index,dtype=int)
    return index
def same_atom(atom_1,atom_2,judge=1e-5):
    natom=len(atom_1)
    if len(atom_1)!=len(atom_2):
        raise Exception("Wrong!, two atom list must be same number")
    index=True
    for i in range(natom):
        if np.min(np.linalg.norm(atom_1-atom_2[i],axis=1))>judge:
            index=False
    return index   

def gen_hamR_new(min_R,max_R,dim_k,dim_r,per):
    n_R=(max_R-min_R+5).prod()
    new_hamR=np.zeros((n_R,dim_r),dtype=int)
    inputs=np.zeros(dim_r,dtype=int)
    per=np.array(per,dtype=int)
    (new_hamR,i0)=gen_R(min_R,max_R,dim_k,per,0,inputs,0,new_hamR)
    for i in range(dim_r): 
        if i==0:
            index=(new_hamR[:,i]>=0)
        else:
            index=(new_hamR[:,i]>0)+(new_hamR[:,i]==0)*(index)
    new_hamR=new_hamR[index]  #删除多余的hopping, 保留不重复的, 因为 R 和 -R 存在重复, 这里是为了减少内存,防止弄混
    index=np.logical_not((new_hamR==0).all(axis=1))
    new_hamR=new_hamR[index]
    a=[np.zeros(dim_r,dtype=int)]
    new_hamR=np.insert(new_hamR,0,a,axis=0)
    return new_hamR

@jit(nopython=True)
def gen_R(min_R,max_R,dim_k,per,k,inputs,i0,new_hamR): #用递归的方法生成可能的 hamR
    if k<dim_k-1:
        for i in range(min_R[k]-2,max_R[k]+3):
            inputs[per[k]]=i
            (new_hamR,i0)=gen_R(min_R,max_R,dim_k,per,k+1,inputs,i0,new_hamR)
        return (new_hamR,i0)
    else:
        for i in range(min_R[k]-2,max_R[k]+3):
            inputs[per[k]]=i
            new_hamR[i0]=inputs
            i0+=1
        return (new_hamR,i0)

def gen_hamR(min_R,max_R,dim_k,dim_r,per): #被 make_supercell_new 调用
    new_hamR=[]
    if dim_k==1:
        for i in range(min_R-2,max_R+3):
            a=np.zeros(dim_r,dtype=int)
            a[per[0]]=i
            new_hamR.append(a.tolist())
        a=np.zeros(dim_r,dtype=int).tolist()
        del new_hamR[new_hamR.index(a)]
        new_hamR.insert(0,a)
    elif dim_k==2:
        for i in range(min_R[0]-2,max_R[0]+3):
            for j in range(min_R[1]-2,max_R[1]+3):
                a=np.zeros(dim_r,dtype=int)
                a[per]=np.array([i,j],dtype=int)
                new_hamR.append(a.tolist())
        a=np.zeros(dim_r,dtype=int).tolist()
        del new_hamR[new_hamR.index(a)]
        new_hamR.insert(0,a)
    elif dim_k==3:
        for i in range(min_R[0]-2,max_R[0]+3):
            for j in range(min_R[1]-2,max_R[1]+3):
                for k in range(min_R[2]-2,max_R[2]+3):
                    a=np.zeros(dim_r,dtype=int)
                    a[per]=np.array([i,j,k],dtype=int)
                    new_hamR.append(a.tolist())
        a=np.zeros(dim_r,dtype=int).tolist()
        del new_hamR[new_hamR.index(a)]
        new_hamR.insert(0,a)
    else:
        raise Exception("Wrong, the k_dim is larger than 3 or less than 0")
    new_hamR=np.array(new_hamR,dtype=int) #产生可能存在 hopping 的 R
    for i in range(new_hamR.shape[1]): 
        if i==0:
            index=new_hamR[:,i]>=0
        else:
            index+=(new_hamR[:,i]>0)+(new_hamR[:,i]==0)*index
    new_hamR=new_hamR[index]
    return new_hamR

def gen_new_structure(min_range,max_range,dim_r,atom0,atom_position,orb0,U_inv): #被 make_supercell_new 调用
    new_orb=[]
    new_atom_position=[]
    new_atom=[]
    new_orb_list=[]
    #产生超胞的原子位置和orbit 位置
    if dim_r==1:
        for i in range(min_range[0],max_range[0]+1):
            for o,orb in enumerate(orb0):
                new_orb.append(np.dot(orb,U_inv)+np.dot([i],U_inv))
                new_orb_list.append(o)
            for a,atom in enumerate(atom_position):
                new_atom_position.append(np.dot(atom,U_inv)+np.dot([i],U_inv))
                new_atom.append(atom0[a])
    elif dim_r==2:
        for i in range(min_range[0],max_range[0]+1):
            for j in range(min_range[1],max_range[1]+1):
                for o,orb in enumerate(orb0):
                    new_orb.append(np.dot(orb,U_inv)+np.dot([i,j],U_inv))
                    new_orb_list.append(o)
                for a,atom in enumerate(atom_position):
                    new_atom_position.append(np.dot(atom,U_inv)+np.dot([i,j],U_inv))
                    new_atom.append(atom0[a])
    elif dim_r==3:
        for i in range(min_range[0],max_range[0]+1):
            for j in range(min_range[1],max_range[1]+1):
                for k in range(min_range[2],max_range[2]+1):
                    for o,orb in enumerate(orb0):
                        new_orb.append(np.dot(orb,U_inv)+np.dot([i,j,k],U_inv))
                        new_orb_list.append(o)
                    for a,atom in enumerate(atom_position):
                        new_atom_position.append(np.dot(atom,U_inv)+np.dot([i,j,k],U_inv))
                        new_atom.append(atom0[a])
    return (new_orb,new_orb_list,new_atom_position,new_atom)

@jit
def gen_orb_list(orb_list,natom,atom):
    for i in range(1,natom):
        orb_list[i]=orb_list[i-1]+atom[i]
    return orb_list

@jit
def gen_orb_match(orb_match,orb_list,atom_match,atom):
    r"""
    这个函数是用来match 轨道之间的对应关系, 返回一个 np.array(dtype=int)的列表. orb_list
    orb_match: 你想要的空的 orb_match
    orb_list: 旧模型的orb_list
    atom_match: 从旧模型到新模型的 list
    atom: 旧模型的 atom
    """
    norb=np.sum(atom)
    natom=len(atom)
    a=0
    for i in range(natom):
        for j in range(atom[i]):
            orb_match[a]=orb_list[atom_match[i]]+j #老模型的orb_list,
            a+=1
    return orb_match

def gen_supercell_hop(new_model,new_hamR,new_orb,new_orb_list,old_ham,old_hamR,old_orb,U):
    #这个是用来被make_supercell 调用的, 用来给超胞 new_model 添加哈密顿量
    new_norb=len(new_orb) #新model 的轨道数量
    old_norb=len(old_orb) #原胞的轨道数量
    for hamR in new_hamR:
        new_hopR=np.zeros((new_norb**2,new_model._dim_r),dtype=int) #新 model 的每一个hopping 对应到原胞的胞间hopping
        ind=np.zeros((new_norb**2,2),dtype=int) #每一个胞间hopping
        ind_old=np.zeros((new_norb**2,2),dtype=int) #原胞 model 的轨道hopping
        (new_hopR,ind,ind_old)=gen_hopR(hamR,new_orb,new_orb_list,old_orb,new_hopR,ind,ind_old,U)
        new_hopR=np.array(new_hopR,dtype=int)
        ind=np.array(ind,dtype=int)
        ind_old=np.array(ind_old,dtype=int)
        if (new_model._hamR==hamR).all(axis=1).any():
            index_R=np.argwhere((new_model._hamR==hamR).all(axis=1))[0][0] 
            use_hopR=new_hopR[0]
            for i in range(new_norb**2):
                orb_i=ind_old[i][0]
                orb_j=ind_old[i][1]
                new_orb_i=ind[i][0]
                new_orb_j=ind[i][1] 
                if np.all(use_hopR==new_hopR[i])==False or (i==0): 
                    use_hopR=new_hopR[i]
                    ind_R=(use_hopR==old_hamR).all(axis=1)
                    ind_R_inv=(-use_hopR==old_hamR).all(axis=1)
                    r_any=ind_R.any()
                    inv_any=ind_R_inv.any()
                if new_model._nspin==1:
                    if r_any:
                        hop=old_ham[ind_R][0][orb_i,orb_j]
                        new_model._ham[index_R][new_orb_i,new_orb_j]=hop
                    elif inv_any:
                        hop=old_ham[ind_R_inv][0].T.conjugate()
                        hop=hop[orb_i,orb_j]
                        new_model._ham[index_R][new_orb_i,new_orb_j]=hop
                else:
                    if r_any:
                        hop=old_ham[ind_R][0][orb_i::old_norb,orb_j::old_norb]
                        new_model._ham[index_R][new_orb_i::new_norb,new_orb_j::new_norb]=hop
                    elif inv_any:
                        hop=old_ham[ind_R_inv][0].conjugate().T
                        hop=hop[orb_i::old_norb,orb_j::old_norb]
                        new_model._ham[index_R][new_orb_i::new_norb,new_orb_j::new_norb]=hop
        elif (new_model._hamR==-hamR).all(axis=1).any():
            index_R=np.argwhere((new_model._hamR==-hamR).all(axis=1))[0][0]
            use_hopR=new_hopR[0]
            for i in range(new_norb**2):
                orb_i=ind_old[i][0]
                orb_j=ind_old[i][1]
                new_orb_i=ind[i][0]
                new_orb_j=ind[i][1]
                if np.all(use_hopR==new_hopR[i])==False or (i==0): 
                    use_hopR=new_hopR[i]
                    ind_R=(use_hopR==old_hamR).all(axis=1)
                    ind_R_inv=(-use_hopR==old_hamR).all(axis=1)
                    r_any=ind_R.any()
                    inv_any=ind_R_inv.any()
                if new_model._nspin==1:
                    if r_any:
                        hop=old_ham[ind_R_inv][0]
                        hop=hop[orb_i,orb_j]
                        new_model._ham[index_R][new_orb_i,new_orb_j]=hop.T.conjugate()
                    elif inv_any:
                        hop=old_ham[ind_R_inv][0]
                        hop=hop[orb_i,orb_j]
                        new_model._ham[index_R][new_orb_i,new_orb_j]=hop
                else:
                    if r_any:
                        hop=old_ham[ind_R_inv][0]
                        hop=hop[orb_i::old_norb,orb_j::old_norb]
                        new_model._ham[index_R][new_orb_i::new_norb,new_orb_j::new_norb]=hop
                    elif inv_any:
                        hop=old_ham[ind_R_inv][0].conjugate().T
                        hop=hop[orb_i::old_norb,orb_j::old_norb]
                        new_model._ham[index_R][new_orb_i::new_norb,new_orb_j::new_norb]=hop.conjugate().T
        else:
            use_hopR=new_hopR[0]
            if new_model._nspin==1:
                use_ham=np.zeros((new_norb,new_norb),dtype=complex)
            if new_model._nspin==2:
                use_ham=np.zeros((new_norb*2,new_norb*2),dtype=complex)
            for i in range(new_norb**2):
                orb_i=ind_old[i][0]
                orb_j=ind_old[i][1]
                new_orb_i=ind[i][0]
                new_orb_j=ind[i][1]
                if np.all(use_hopR==new_hopR[i])==False or (i==0): 
                    use_hopR=new_hopR[i]
                    ind_R=(use_hopR==old_hamR).all(axis=1)
                    ind_R_inv=(-use_hopR==old_hamR).all(axis=1)
                    r_any=ind_R.any()
                    inv_any=ind_R_inv.any()
                if new_model._nspin==1:
                    if r_any:
                        hop=old_ham[ind_R][0][orb_i,orb_j]
                        use_ham[new_orb_i,new_orb_j]=hop
                    elif inv_any:
                        hop=old_ham[ind_R_inv][0][orb_j,orb_i].conjugate()
                        use_ham[new_orb_i,new_orb_j]=hop
                else:
                    if r_any:
                        hop=old_ham[ind_R][0][orb_i::old_norb,orb_j::old_norb]
                        use_ham[new_orb_i::new_norb,new_orb_j::new_norb]=hop
                    elif inv_any:
                        hop=old_ham[ind_R_inv][0].conjugate().T
                        hop=hop[orb_i::old_norb,orb_j::old_norb]
                        use_ham[new_orb_i::new_norb,new_orb_j::new_norb]=hop
            new_model._ham=np.append(new_model._ham,[use_ham],axis=0)
            new_model._hamR=np.append(new_model._hamR,[hamR],axis=0)
    return new_model

@jit(nopython=True)
def gen_hopR_1(hamR,new_orb,hop_R_1,ind): #gen_hopR_1 的前置步骤
    new_orb_i=0
    a=0
    for i,new_orb_i in enumerate(new_orb):
        for j,new_orb_j in enumerate(new_orb):
            hop_R_1[a]=hamR+new_orb_j-new_orb_i
            ind[a]=[i,j]
            a+=1
    return (hop_R_1,ind)

@jit(nopython=True)
def gen_hopR_2(old_orb,new_orb_list,ind,hop_R,ind_old): #gen_hopR 的后置步骤
    n_R=len(hop_R)
    for i in range(n_R):
        orb_i=new_orb_list[ind[i][0]]
        orb_j=new_orb_list[ind[i][1]]
        hop_R[i]+=-old_orb[orb_j]+old_orb[orb_i]
        ind_old[i]=[orb_i,orb_j]
    return (hop_R,ind_old)

def gen_hopR(hamR,new_orb,new_orb_list,old_orb,new_hopR,ind,ind_old,U): #这个程序是用来产生可能的hop, 被 make_supercell 间接调用
    norb=len(new_orb)
    dim_r=hamR.shape[0]
    hop_R_1=np.zeros((norb**2,dim_r),dtype=float)
    ind=np.zeros((norb**2,2),dtype=int)
    ind_old=np.zeros((norb**2,2),dtype=int)
    (hop_R_1,ind)=gen_hopR_1(hamR,new_orb,hop_R_1,ind)
    hop_R_1=np.dot(hop_R_1,U)
    (hop_R_2,ind_old)=gen_hopR_2(old_orb,new_orb_list,ind,hop_R_1,ind_old)
    hop_R_2=shift_to_zero(hop_R_2)
    hop_R=np.array(hop_R_2,dtype=int)
    return(hop_R,ind,ind_old)

@jit
def get_list(dim,n_min,n_max):
    ss=[]
    if dim==1:
        for i in range(n_min,n_max):
            ss.append(np.array([i]))
    elif dim==2:
        for i in range(n_min,n_max):
            for j in range(n_min,n_max):
                ss.append(np.array([i,j]))
    elif dim==3:
        for i in range(n_min,n_max):
            for j in range(n_min,n_max):
                for k in range(n_min,n_max):
                    ss.append(np.array([i,j,k]))
    elif dim==4:
        for i in range(n_min,n_max):
            for j in range(n_min,n_max):
                for k in range(n_min,n_max):
                    for l in range(n_min,n_max):
                        ss.append(np.array([i,j,k,l]))
    else:
        raise Exception("dim only can be 1,2,3,4")
    return ss

def gen_mesh_arr(mesh_arr,dim_arr,startk,usek=None,r=0):
    k=[]
    if r==0:
        for i in range(mesh_arr[r]-1):
            usek=np.zeros(dim_arr,dtype=float)
            usek[r]=startk[r]+float(i)/float(mesh_arr[r]-1)
            k0=gen_mesh_arr(mesh_arr,dim_arr,startk,usek,r+1)
            k.extend(k0)
        return k
    elif r<dim_arr-1:
        for i in range(mesh_arr[r]-1):
            usek[r]=startk[r]+float(i)/float(mesh_arr[r]-1)
            k0=gen_mesh_arr(mesh_arr,dim_arr,startk,usek,r+1)
            k.extend(k0)
        return k 
    else:
        for i in range(mesh_arr[r]-1):
            usek[r]=startk[r]+float(i)/float(mesh_arr[r]-1)
            k.append(usek)
            k=copy.deepcopy(k)
        return k

@njit('f8[:](f8[:],f8[:],f8[:],f8)',fastmath=True)
def Gauss(x,y,center,sigma):
    for i in range(center.shape[0]):
        y+=1/np.sqrt(2*np.pi)/sigma*np.exp(-(x-center[i])**2/(2*sigma**2))
    y*=1/center.shape[0]
    return y

def calculate_omegan(J,v,evec,band,nsta,omega_n,og,eta):
    nk=band.shape[0]
    A1=evec.conj()@(J@evec.transpose(0,2,1))
    A2=evec.conj()@(v@evec.transpose(0,2,1))
    U0=np.zeros((nk,nsta,nsta),dtype=complex)
    for i in range(nsta):
        for j in range(nsta):
            U0[:,i,j]=1/((band[:,i]-band[:,j])**2-(og+1.j*eta)**2)
    #omega_n=-2*np.einsum("kij,kji,kij->ki",A1,A2,U0).imag
    omega_n=-2*np.diagonal((A1*U0)@A2,axis1=1,axis2=2).imag
    return omega_n

def gen_kvector(lat):
    if len(lat)==3:
        K=(2*np.pi)/(np.linalg.det(_lat))*np.array([np.cross(lat[1],lat[2]),np.cross(lat[2],lat[0]),np.cross(lat[0],lat[1])])
    elif len(lat)==2:
        array_1=self._lat[1][[1,0]]
        array_2=self._lat[0][[1,0]]
        array_1[1]*=-1
        K=(2*np.pi)*np.array([array_1,array_2])
    return K

def gen_ham_try(ham0,useham,ind_R,orb,spin,k_point):
    newham=np.einsum("s,sij->ij",np.exp(2.j*np.pi*np.dot(ind_R,k_point)),useham,optimize=True)
    newham+=newham.conj().T+ham0
    U=np.diag(np.exp(2.j*np.pi*np.dot(orb,k_point)))
    if spin==2:
        U=np.kron([[1,0],[0,1]],U)
    newham=np.dot(newham,U)
    newham=np.dot(U.T.conj(),newham)
    return newham


def gen_ham_v(ham0,useham,ind_R,orb,ind_R0,U0,spin,k_point):
    U=np.exp(2.j*np.pi*np.dot(orb,k_point))
    if spin==2:
        U=np.append(U,U)
    U=np.diag(U)
    Us=np.exp(2.j*np.pi*np.dot(ind_R,k_point))
    newham0=np.einsum("s,sij->ij",Us,useham,optimize=True)
    newham0+=ham0+newham0.conj().T
    newham0=U0*newham0
    ham1=np.einsum("s,sij,sr->rij",Us,useham,ind_R0*1.j,optimize='greedy')
    ham1+=ham1.conj().transpose(0,2,1)
    ham=newham0+ham1
    ham=np.matmul(ham,U)
    ham=np.matmul(U.T.conjugate(),ham)
    return ham


def gen_r(r0,rmatrix,R,U,k):
    r=np.einsum("s,sij->ij",np.exp(2.j*np.pi*np.dot(R,k)),rmatrix,optimize=True)
    r+=r0+r.T.conj()
    r=U.T.conj()@r@U
    return r
