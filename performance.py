########################################################################
#Script to plot and extract data related to airplane performance.
########################################################################

#Imports
import numpy, sys, os, pdb, scipy.optimize, argparse
import matplotlib.pyplot as plt
sys.path.append('../utils')
import warnings
warnings.filterwarnings('ignore')
from standard_atmosphere import *
from read_data import *
from write_file import *

########################################################################

#Classes and functions:

class Airplane:
    '''Contains all necessary gemetric and aerodynamic data for one airplane.'''
    def __init__(self,ac_name,W,S,df,b,c,sweep,theta,clmax,nlim,P0_eng,sfc,propulsion,M,h,h_ref,V,VA,Vcruise,cd0,fuel_W,output,f,eta_p=1,supercharger=0):
        self.f = f #output file instance class
        self.ac_name = ac_name
        self.__W = W
        self.__S = S
        self.__df = df #fuselage diameter at wing root
        self.__df_over_b = df/b
        self.__b = b
        self.__c = c #mean aerodynamic chord
        self.__sweep = sweep*numpy.pi/180 # 1/4 c sweep, converted from deg to rad
        self.__AR = b**2/S
        self.f.header('Aditional information')
        self.f.write_number('Aspect ratio',self.__AR,' ')
        self.__clmax = clmax
        self.__nlim = nlim #limit maneuvre load factor (flight envelope)
        if propulsion == 'propeller':
            self.__eta_p = eta_p #eficiency
            self.__P0_eng = P0_eng*eta_p #Sea level engine power(propeller)
            self.__supercharger = supercharger
        else:
            self.__T0_eng = P0_eng #Sea level engine thrust(jet)
        self.__propulsion = propulsion
        self.__sfc = sfc #specific fuel consumption
        self.__fuel_W = fuel_W
        # The next parameters without double underscore are the range to plot/run 
        self.M = numpy.linspace(0,M,100)
        self.h = numpy.linspace(0,h,100) #h in m 
        self.V = numpy.linspace(0,V,100)
        self.Vcruise = Vcruise
        self.rho = numpy.array([density_SI(h) for h in self.h])
        self.Vs = numpy.zeros((len(self.rho),len(clmax)))
        for i in range(len(self.rho)):
            for j in range(len(clmax)):
                self.Vs[i,j] = numpy.sqrt(W/(0.5*clmax[j]*self.rho[i]*S))
        for i in range(len(clmax)):
            if output == 'SI':
                self.f.write_list('Altitude (m)','Stall speed (m/s) condition {}'.format(i+1),numpy.vstack([self.h,self.Vs[:,i]]).T)
            elif output == 'imperial':
                self.f.write_list('Altitude (ft)','Stall speed (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,self.Vs[:,i]/0.514444]).T)
        self.h_ref = h_ref #reference altitude for plotting (e.g. cruise)
        self.VA = VA #maneuvre velocity to plot turning parameters.
        self.theta = theta*numpy.pi/180 #takeoff AoA
        # Drag polar parameters
        self.__cd0 = cd0
        oswald = self.oswald_shevell()
        w_efficiency = 2/(2-self.__AR+numpy.sqrt(4+self.__AR**2*(1+numpy.tan(self.__sweep)**2)))
        self.__k = 1/(numpy.pi*self.__AR*w_efficiency)*numpy.ones_like(cd0)
        for i in range(len(clmax)):
            self.f.write_number('Oswald number cond {}'.format(i+1),oswald[i],' ')
        self.f.write_number('Wing efficiency',w_efficiency,' ')
        for i in range(len(clmax)):
            self.f.write_number('Drag polar k cond {}'.format(i+1),self.__k[i],' ')
        # should replace it by the proper curve later:
        self.cl = numpy.zeros((110,len(clmax)))
        for j,cl in enumerate(clmax):
            self.cl[:,j] = numpy.hstack([numpy.linspace(0,cl,100),numpy.linspace(cl,clmax[j]*0.9,10)])
        self.cd = cd0 + self.__k*self.cl**2
        for i in range(90,110):
            self.cd[i,:] = cd0 + self.__k*self.cl[i,:]**2 + clmax*(self.cl[i,:]-0.9*clmax)**2
        for i in range(100,110):
            self.cd[i,:] += 2*(self.cd[100,:]-self.cd[i,:])
        for i in range(len(clmax)):
            self.f.write_list('Cl','Cd condition {}'.format(i+1),numpy.vstack([self.cl[:,i],self.cd[:,i]]).T)
        # Max aerodynamic efficiency
        self.__Emax = 1/(2*numpy.sqrt(self.__k*cd0))
        self.__cl_star = numpy.sqrt(cd0/self.__k) # cl for max efficiency
        for i in range(len(clmax)):
            self.f.write_number('Max efficiency con {}'.format(i+1),self.__Emax[i],' ')
        for i in range(len(clmax)):
            self.f.write_number('Cl for max E cond {}'.format(i+1),self.__cl_star[i],' ')
        self.output = output #output format (units)

    '''Methods to calculate the performance data.'''
    def oswald_shevell(self):
        '''Estimates the oswald factor based on Shevell's method.'''
        s = 1-1.9316*self.__df_over_b**2
        k = 0.375*self.__cd0
        # sweep 1/4 c in rad
        sweep_corr = 1-0.02*self.__AR**0.7*self.__sweep**2.2
        # u = planform efficiency, considered 1
        u = numpy.ones_like(self.__cd0)
        e = sweep_corr/(numpy.pi*self.__AR*k+1/(u*s))
        return e

    def specific_fuel_consuption(self,V,M,h):
        '''Returns the specific fuel consuption for different types of engine and flight conditions.'''
        propulsion = self.__propulsion
        if propulsion == 'propeller':
            self.__sfc = sfc #specific fuel consumption
        elif propulsion == 'ramjet':
            self.__tsfc = sfc
        elif propulsion == 'turbojet':
            self.__tsfc = sfc
            # it should depend on M => TSFC = 1 +k*M, but k is empirical. M<1
        elif propulsion == 'turbofan':
            self.__ct = sfc
            # it should depend on M => ct = B*(1 +k*M), but B and k are empirical.        

    def engine_power_thrust(self,h):
        '''Calculates the engine power for propeller systems or thrust for jet engines for different flight conditions. h in ft.'''
        h /= 0.3048 # convert meter to feet.
        try:
            P0 = self.__P0_eng
        except:
            T0 = self.__T0_eng
        if self.__propulsion == 'propeller':
            if not self.__supercharger:
                P = P0*(1.132*density_imperial(h)/density_imperial(0)-0.132) #see Roskan
            else:
                P = P0
        #Fix this later
        elif self.__propulsion == 'ramjet':
            T = T0*(density_imperial(h)/density_imperial(0))
        elif self.__propulsion == 'turbojet':
            T = T0*(density_imperial(h)/density_imperial(0))
        elif self.__propulsion == 'turbofan':
            T = T0*(density_imperial(h)/density_imperial(0))
        elif self.__propulsion == 'glider':
            P = 0.0
        try:
            return P
        except:
            return T
    def steady_propeller(self,V,h,j):
        '''Equation to get the range of velocities the airplane can flight at certain altitude.'''
        Pd = self.engine_power_thrust(h)
        return 0.5*density_SI(h)*self.__S*self.__cd0[j]*V**3+(2*self.__k[j]*self.__W**2/(density_SI(h)*self.__S*V)) - Pd

    def steady_level_flight(self,path):
        '''Drag, required thrust/power for the engine and absolute ceiling calculations based on the steady level flight equations solution.'''
        # minimum drag velocity, also known as reference velocity
        steady_path = os.path.join(path,'steady')
        if not os.path.isdir(steady_path):
            os.mkdir(steady_path)
        self.f.header('Steady flight')
        self.VR = numpy.zeros((len(self.rho),len(self.__k)))
        for i in range(len(self.rho)):
            for j in range(len(self.__k)):
                self.VR[i,j] = numpy.sqrt(2*self.__W/(self.rho[i]*self.__S))*numpy.power(self.__k[j]/self.__cd0[j],0.25)
        for i in range(len(self.__k)):
            if self.output == 'SI':
                self.f.write_list('Altitude (m)','Reference speed (m/s) condition {}'.format(i+1),numpy.vstack([self.h,self.VR[:,i]]).T)
            elif self.output == 'imperial':
                self.f.write_list('Altitude (ft)','Reference speed (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,self.VR[:,i]/0.514444]).T)
        if self.__propulsion == 'propeller':
            # minimum required power parameters:
            VPmin = self.VR/numpy.power(3,0.25)
            PRmin = numpy.zeros_like(self.VR)
            for i in range(len(self.rho)):
                for j in range(len(self.__k)):
                    PRmin[i,j] = (self.rho[i]**2*self.__S**2*self.__cd0[j]*self.VR[i,j]**4 + 12*self.__k[j]*self.__W**2)/(2*numpy.power(27,0.25)*self.rho[i]*self.__S*self.VR[i,j])
            clPmin = numpy.sqrt(3)*self.__cl_star
            EPmin = numpy.sqrt(3)/2*self.__Emax
            # for the reference altitude:
            V_nl = numpy.sqrt(2*self.__W/(density_SI(self.h_ref)*self.__S*(self.cl[10:,:]))) #range of speed that correspont to the nonlinear case, removing low cl cases.
            PR_nl = 0.5*density_SI(self.h_ref)*self.__S*V_nl**3*self.cd[10:,:] #considering stall 
            VS = numpy.zeros(len(self.__k))
            V_ref = numpy.zeros((100,len(self.__k)))
            for i in range(len(self.__k)):
                VS[i] = numpy.min(V_nl[:,i])
                V_ref[:,i] = numpy.linspace(0.8*VS[i],numpy.max(self.V),100)
            PD = numpy.ones_like(self.V)*self.engine_power_thrust(self.h_ref)
            PR = numpy.zeros_like(V_ref)
            for i in range(V_ref.shape[0]):
                for j in range(len(self.__k)):
                    PR[i,j] = 0.5*density_SI(self.h_ref)*self.__S*V_ref[i,j]**3*self.__cd0[j]+2*self.__k[j]*self.__W**2/(density_SI(self.h_ref)*V_ref[i,j]*self.__S)
            #Solution for Vmax and Vmin:
            sol = numpy.zeros((len(self.h),2,len(self.__k)))
            for j in range(len(self.__k)):
                sol1 = []
                for i,h in enumerate(self.h):
                   #if i == 0:
                        ans_min = scipy.optimize.ridder(self.steady_propeller,0,self.V[-1],args=(h,j))
                        ans_max = scipy.optimize.ridder(self.steady_propeller,ans_min,self.V[-1],args=(h,j))
                        sol1.append([ans_min,ans_max])
                   #else:
                   #    # change initial guess until solution is correct
                   #    x0 = sol1[-1]
                   #    err1 = 1
                   #    err2 = 1
                   #    n = 0
                   #    while err1 > 0.2 or err2 > 0.2:
                   #        ans = scipy.optimize.root(self.steady_propeller,x0,args=(h,j))
                   #        err1 = numpy.absolute(ans.x[0]-sol1[-1][0])/sol1[-1][0]
                   #        err2 = numpy.absolute(ans.x[1]-sol1[-1][1])/sol1[-1][1]
                   #        if err1 > 0.2:
                   #            x0[0] += 1
                   #        if err2 > 0.2:
                   #            x0[1] += 1
                   #        n += 1
                   #        if n == 20:
                   #            
                   #            break
                   #    sol1.append(ans.x)
                sol[:,:,j] = numpy.array(sol1)
            abs_ceil = numpy.zeros_like(self.__k)
            ind_ceil = numpy.zeros_like(self.__k)
            for k in range(len(self.__k)):
                for i in xrange(2,sol.shape[0]):
                    if sol[i,0,k] +(sol[i,0,k]-sol[i-1,0,k]) > sol[i-1,1,k]: #the lower curve will be above the upper one.
                        abs_ceil[k] = self.h[i+1]
                        ind_ceil[k] = i+1
                        break
                    else:
                        abs_ceil[k] = self.h[-1]
                        ind_ceil[k] = -1
            plt.figure()
            if self.output == 'SI':
                plt.plot(self.h,VPmin)
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Minimum power speed (m/s) condition {}'.format(i+1),numpy.vstack([self.h,VPmin[:,i]]).T)
                plt.plot(self.h,self.VR)
            elif self.output == 'imperial':
                plt.plot(self.h/0.3048,VPmin/0.514444)
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Minimum power speed (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,VPmin[:,i]/0.514444]).T)
                plt.plot(self.h/0.3048,self.VR/0.514444)
            plt.title('Relevant steady condition velocities.')
            if self.output == 'SI':
                plt.xlabel('Altitude (m)')
                plt.ylabel('V (m/s)')
            elif self.output == 'imperial':
                plt.xlabel('Altitude (ft)')
                plt.ylabel('V (knots)')
            legend = []
            for i in range(len(self.__k)):
                legend.append('VPmin cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('VR cond {}'.format(i+1))
            plt.legend(legend,loc='upper left')
            plt.grid()
            plt.savefig(os.path.join(steady_path,'Steady velocities.png'))
            plt.close()
            plt.figure()
            if self.output == 'SI':
                plt.plot(self.h,PRmin)
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Minimum required power (W) condition {}'.format(i+1),numpy.vstack([self.h,PRmin[:,i]]).T)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Minimum required power (W)')
            elif self.output == 'imperial':
                for i in range(PRmin.shape[0]):
                    for j in range(len(self.__k)):
                        PR_var = Variable([PRmin[i,j],'N*m/s'])
                        PR_var.convert2imperial()
                        PRmin[i,j] = PR_var.value*0.001818 #convert lbs*ft/s to hp
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Minimum required power (HP) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,PRmin[:,i]]).T)
                plt.plot(self.h/0.3048,PRmin)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Minimum required power (HP)')
            legend = []
            for i in range(len(self.__k)):
                legend.append('Cond {}'.format(i+1))
            plt.legend(legend,loc='upper left')
            plt.grid()
            plt.savefig(os.path.join(steady_path,'PRmin.png'))
            plt.close()
            for i in range(len(self.__k)):
                plt.figure()
                plt.title('Required and available power at ref altitude.')
                if self.output == 'SI':
                    plt.plot(V_ref[:,i],PR[:,i])
                    plt.plot(V_nl[:,i],PR_nl[:,i],'.')
                    plt.plot(self.V,PD)
                    plt.xlabel('Altitude (m)')
                    plt.ylabel('Engine Power (W)')
                elif self.output == 'imperial':
                    plt.plot(V_ref[:,i]/0.514444,PR[:,i]*0.0013410220888)
                    plt.plot(V_nl[:,i]/0.514444,PR_nl[:,i]*0.0013410220888,'.')
                    plt.plot(self.V/0.514444,PD*0.0013410220888)
                    plt.xlabel('Altitude (ft)')
                    plt.ylabel('Engine Power (HP)')
                plt.legend(['Required Power','Required Power considering stall','Available Power'],loc='upper left')
                plt.grid()
                plt.savefig(os.path.join(steady_path,'PD_and_PR_cond{}.png'.format(i+1)))
                plt.close()
            for i in range(len(self.__k)):
                plt.figure()
                plt.title('Operational velocities.')
                if self.output == 'SI':
                    plt.plot(self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],0,i])
                    plt.plot(self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],1,i])
                    plt.plot(self.h[0:ind_ceil[i]],self.Vs[0:ind_ceil[i],i])
                    self.f.write_list('Altitude (m)','Minimum velocity (m/s) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],0,i]]).T)
                    self.f.write_list('Altitude (m)','Maximum velocity (m/s) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],1,i]]).T)
                    plt.xlabel('Altitude (m)')
                    plt.ylabel('Velocity (m/s)')
                    plt.legend(['Minimum','Maximum','Stall'])
                elif self.output == 'imperial':
                    plt.plot(self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],0,i]/0.514444)
                    plt.plot(self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],1,i]/0.514444)
                    plt.plot(self.h[0:ind_ceil[i]]/0.3048,self.Vs[0:ind_ceil[i],i]/0.514444)
                    self.f.write_list('Altitude (ft)','Minimum velocity (knots) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],0,i]/0.514444]).T)
                    self.f.write_list('Altitude (ft)','Maximum velocity (knots) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],1,i]/0.514444]).T)
                    plt.xlabel('Altitude (ft)')
                    plt.ylabel('Velocity (knots)')
                    plt.legend(['Minimum','Maximum','Stall'])
                plt.grid()
                plt.savefig(os.path.join(steady_path,'V_operation_cond{}.png'.format(i+1)))
            plt.close()
            for i in range(len(self.__k)):
                self.f.write_number('Cl min power cond {}'.format(i+1),clPmin[i],' ')
                self.f.write_number('E min power cond {}'.format(i+1),EPmin[i],' ')
                if self.output == 'SI':
                    self.f.write_number('Abs ceiling cond {}'.format(i+1),abs_ceil[i],'m')
                elif self.output == 'imperial':
                    self.f.write_number('Abs ceiling cond {}'.format(i+1),abs_ceil[i]/0.3048,'ft')
        elif self.__propulsion in ['turbojet','turbofan','ramjet']:
            # Minimum drag
            Dmin = 2*self.__W*numpy.sqrt(self.__k*self.__cd0)
            TRmin = Dmin
            # for the reference altitude:
            V_nl = numpy.sqrt(2*self.__W/(density_SI(self.h_ref)*self.__S*(self.cl[10:]))) #range of speed that correspont to the nonlinear case, removing low cl cases
            TR_nl = 0.5*density_SI(self.h_ref)*self.__S*V_nl**2*self.cd[10:] #considering stall
            VS = numpy.zeros(len(self.__k))
            V_ref = numpy.zeros((100,len(self.__k)))
            for i in range(V_ref.shape[0]):
                for j in range(len(self.__k)):
                    VS[j] = numpy.min(V_nl[:,j])
                    V_ref[:,j] = numpy.linspace(0.8*VS[j],numpy.max(self.V),100)
            TD = numpy.ones_like(self.V)*self.engine_power_thrust(self.h_ref)
            TR = numpy.zeros_like(V_ref)
            for i in range(V_ref.shape[0]):
                for j in range(len(self.__k)):
                    TR[i,j] = 0.5*density_SI(self.h_ref)*self.__S*V_ref[i,j]**2*self.__cd0[j]+2*self.__k[j]*self.__W**2/(density_SI(self.h_ref)*V_ref[i,j]**2*self.__S)
            #Solution for Vmax and Vmin:
            sol = numpy.zeros((len(self.h),2,len(self.__k)))
            ind_ceil = numpy.zeros(len(self.__k))
            abs_ceil = numpy.zeros(len(self.__k))
            for j in range(len(self.__k)):
                for i,h in enumerate(self.h):
                    A = 0.5*self.rho[i]*self.__S*self.__cd0[j]
                    B = 2*self.__k[j]*self.__W**2/(self.rho[i]*self.__S)
                    T = self.engine_power_thrust(h)
                    if T**2-4*A*B >= 0:
                        sol[i,:,j] = [numpy.sqrt((T-numpy.sqrt(T**2-4*A*B))/(2*A)),numpy.sqrt((T+numpy.sqrt(T**2-4*A*B))/(2*A))]
                    else:
                        sol[i,:,j] = [numpy.sqrt((T)/(2*A)),numpy.sqrt((T)/(2*A))]
                for i in xrange(len(self.h)):
                    if sol[i,1,j]<=sol[i,0,j]:
                        ind_ceil[j] = i+1
                        abs_ceil[j] = self.h[i+1]
                        break
                    elif i == len(self.h)-1:
                        ind_ceil[j] = i
                        abs_ceil[j] = self.h[i]
            for i in range(len(self.__k)):
                plt.figure()
                plt.title('Operational velocities.')
                if self.output == 'SI':
                    plt.plot(self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],0,i])
                    plt.plot(self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],1,i])
                    plt.plot(self.h[0:ind_ceil[i]],self.Vs[0:ind_ceil[i],i])
                    self.f.write_list('Altitude (m)','Minimum velocity (m/s) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],0,i]]).T)
                    self.f.write_list('Altitude (m)','Maximum velocity (m/s) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]],sol[0:ind_ceil[i],1,i]]).T)
                    plt.xlabel('Altitude (m)')
                    plt.ylabel('Velocity (m/s)')
                    plt.legend(['Minimum','Maximum','Stall'])
                elif self.output == 'imperial':
                    plt.plot(self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],0,i]/0.514444)
                    plt.plot(self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],1,i]/0.514444)
                    plt.plot(self.h[0:ind_ceil[i]]/0.3048,self.Vs[0:ind_ceil[i],i]/0.514444)
                    self.f.write_list('Altitude (ft)','Minimum velocity (knots) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],0,i]/0.514444]).T)
                    self.f.write_list('Altitude (ft)','Maximum velocity (knots) condition {}'.format(i+1),numpy.vstack([self.h[0:ind_ceil[i]]/0.3048,sol[0:ind_ceil[i],1,i]/0.514444]).T)
                    plt.xlabel('Altitude (ft)')
                    plt.ylabel('Velocity (knots)')
                    plt.legend(['Minimum','Maximum','Stall'])
                plt.grid()
                plt.savefig(os.path.join(steady_path,'V_operation_cond{}.png'.format(i+1)))
            plt.close()
            for i in range(len(self.__k)):
                plt.figure()
                plt.title('Requiered and available thrust at ref altitude.')
                if self.output == 'SI':
                    plt.plot(V_ref[:,i],TR[:,i])
                    plt.plot(V_nl[:,i],TR_nl[:,i])
                    plt.plot(self.V,TD)
                    plt.xlabel('Velocity (m/s)')
                    plt.ylabel('Thrust (N)')
                elif self.output == 'imperial':
                    plt.plot(V_ref[:,i]/0.514444,TR[:,i]*0.224809)
                    plt.plot(V_nl[:,i]/0.514444,TR_nl[:,i]*0.224809)
                    plt.plot(self.V/0.514444,TD*0.224809)
                    plt.xlabel('Velocity (knots)')
                    plt.ylabel('Thrust (lbs)')
                plt.legend(['Required Thrust','Required Thrust considering stall','Available Thrust'],loc='upper left')
                plt.grid()
                plt.savefig(os.path.join(steady_path,'TD_and_TR_cond{}.png'.format(i+1)))
                plt.close()
            for i in range(len(self.__k)):
                if self.output == 'SI':
                    self.f.write_number('TR min cond {}'.format(i+1),TRmin[i],'N')
                    self.f.write_number('Abs ceiling cond {}'.format(i+1),abs_ceil[i],'m')
                elif self.output == 'imperial':
                    self.f.write_number('TR min cond {}'.format(i+1),TRmin[i]*0.224809,'lbs')
                    self.f.write_number('Abs ceiling cond {}'.format(i+1),abs_ceil[i]/0.3048,'ft')
        return
    def unpowered_flight(self,path):
        '''Fundamental parameters for gliders and airplanes with inoperant engines.'''
        unpowered_path = os.path.join(path,'unpowered')
        if not os.path.isdir(unpowered_path):
            os.mkdir(unpowered_path)
        self.f.header('Unpowered flight')
        #Minimum gliding angle. gama=arctan(-cd/cl)
        gama = -numpy.arctan(self.cd/self.cl)
        gama_min = -1/self.__Emax #rad
        gama_min *= 180/numpy.pi #deg
        Vgama_min = self.VR
        #Minimum Rate-of-descent:
        RD = numpy.sqrt(2*self.__W/(density_SI(self.h_ref)*self.__S*(self.cl*numpy.cos(gama)-self.cd*numpy.sin(gama))))*numpy.sin(gama)
        RD_min = numpy.zeros_like(self.VR)
        for j in range(len(self.__k)):
            for i,h in enumerate(self.h):
                RD_min[i,j] = numpy.sqrt(32*self.__W/(3*self.rho[i]*self.__S))*numpy.power(self.__k[j]**3*self.__cd0[j]/3,0.25)
        for i in range(len(self.__k)):
            if self.output == 'SI':
                self.f.write_list('Altitude (m)','V Min gliding angle (m/s) condition {}'.format(i+1),numpy.vstack([self.h,RD_min[:,i]]).T)
            elif self.output == 'imperial':
                self.f.write_list('Altitude (ft)','Min rate-of-descent (ft/s) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,RD_min[:,i]/0.3048]).T)
        
        VRD_min = self.VR/numpy.power(3,0.25)
        #Unpowered dive speed:
        VD = numpy.zeros_like(self.VR)
        for j in range(len(self.__k)):
            for i,h in enumerate(self.h):
                VD[i,j] = numpy.sqrt(2*self.__W/(self.rho[i]*self.__S*self.__cd0[j]))
        plt.figure()
        plt.title('Relevant velocities for unpowered flight')
        legend = []
        if self.output == 'SI':
            plt.plot(self.h,self.VR)
            for i in range(len(self.__k)):
                legend.append('Minimum gliding angle cond {}'.format(i+1))
                self.f.write_list('Altitude (m)','V Min gliding angle (m/s) condition {}'.format(i+1),numpy.vstack([self.h,self.VR[:,i]]).T)
                self.f.write_list('Altitude (m)','V Min rate-of-descent (m/s) condition {}'.format(i+1),numpy.vstack([self.h,VRD_min[:,i]]).T)
                self.f.write_list('Altitude (m)','Dive speed (m/s) condition {}'.format(i+1),numpy.vstack([self.h,VD[:,i]]).T)
            plt.plot(self.h,VRD_min)
            for i in range(len(self.__k)):
                legend.append('Minimum Rate-of-descent cond {}'.format(i+1))
            plt.plot(self.h,VD)
            for i in range(len(self.__k)):
                legend.append('Dive speed cond {}'.format(i+1))
            plt.xlabel('Altitude (m)')
            plt.ylabel('Velocity (m/s)')
        elif self.output == 'imperial':
            plt.plot(self.h/0.3048,self.VR/0.514444)
            for i in range(len(self.__k)):
                legend.append('Minimum gliding angle cond {}'.format(i+1))
                self.f.write_list('Altitude (ft)','V Min gliding angle (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,self.VR[:,i]/0.514444]).T)
                self.f.write_list('Altitude (ft)','V Min rate-of-descent (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,VRD_min[:,i]/0.514444]).T)
                self.f.write_list('Altitude (ft)','Dive speed (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,VD[:,i]/0.514444]).T)
            plt.plot(self.h/0.3048,VRD_min/0.514444)
            for i in range(len(self.__k)):
                legend.append('Minimum Rate-of-descent cond {}'.format(i+1))
            plt.plot(self.h/0.3048,VD/0.514444)
            for i in range(len(self.__k)):
                legend.append('Dive speed cond {}'.format(i+1))
            plt.xlabel('Altitude (ft)')
            plt.ylabel('Velocity (knots)')
        plt.legend(legend,loc='upper left')
        plt.grid()
        plt.savefig(os.path.join(unpowered_path,'Unpowered_velocities.png'))
        plt.close()
        #drag polar
        plt.figure()
        legend = []
        plt.plot(self.cd,self.cl)
        plt.title('{} drag polar'.format(self.ac_name))
        for i in range(len(self.__k)):
            legend.append('Condition {}'.format(i+1))
        plt.xlabel('Cd')
        plt.ylabel('Cl')
        plt.grid()
        plt.savefig(os.path.join(unpowered_path,'Drag_polar.png'))
        plt.close()
        #Hodograph, speed polar
        Vh = numpy.sqrt(2*self.__W/(density_SI(self.h_ref)*self.__S*(self.cl*numpy.cos(gama)-self.cd*numpy.sin(gama))))*numpy.cos(gama) #horizontal velocity
        plt.figure()
        plt.title('{} hodograph'.format(self.ac_name))
        if self.output == 'SI':
            plt.plot(Vh,RD)
            for i in range(len(self.__k)):
                self.f.write_list('Velocity (m/s)','Rate-of-descent (m/s) condition {}'.format(i+1),numpy.vstack([Vh[:,i],RD[:,i]]).T)
            plt.xlabel('Velocity (m/s)')
            plt.ylabel('Rate-of-descent (m/s)')
        elif self.output == 'imperial':
            plt.plot(Vh/0.514444,RD/0.3048)
            for i in range(len(self.__k)):
                self.f.write_list('Velocity (knots)','Rate-of-descent (ft/s) condition {}'.format(i+1),numpy.vstack([Vh[:,i]/0.514444,RD[:,i]/0.3048]).T)
            plt.xlabel('Velocity (knots)')
            plt.ylabel('Rate-of-descent (ft/s)')
        legend = []
        for i in range(len(self.__k)):
            legend.append('Condition {}'.format(i+1))
            self.f.write_number('Gama min cond {}'.format(i+1),gama_min[i],'Deg')
        plt.grid()
        plt.savefig(os.path.join(unpowered_path,'Hodograph.png'))
        plt.close()
        return

    def gama_max_propeller(self,V,h,j):
        '''Function to be solved to calculate the maximum climb angle.'''
        sin_gama = V**4 + self.engine_power_thrust(h)*V/(density_SI(h)*self.__S*self.__cd0[j]*self.__W) - 4*self.__k[j]*(self.__W/self.__S)**2/(density_SI(h)**2*self.__cd0[j])
        return sin_gama

    def climb_performance(self,path):
        '''Check Roskan chap9 for climb performance details.'''
        climb_path = os.path.join(path,'climb')
        if not os.path.isdir(climb_path):
            os.mkdir(climb_path)
        self.f.header('Climb performance')
        #Climb angle for h_ref no stall:
        gama = numpy.zeros_like(self.VR)
        if self.__propulsion == 'propeller':
            for j in range(len(self.__k)):
                for i,V in enumerate(self.V):
                    gama[i,j] = numpy.arcsin((self.engine_power_thrust(self.h_ref)/V-0.5*density_SI(self.h_ref)*self.__S*self.__cd0[j]*V**2-2*self.__k[j]*self.__W**2/(density_SI(self.h_ref)*self.__S*V**2))/self.__W)*180/numpy.pi
        else:
            for j in range(len(self.__k)):
                for i,V in enumerate(self.V):
                    gama[i,j] = numpy.arcsin((self.engine_power_thrust(self.h_ref)-0.5*density_SI(self.h_ref)*self.__S*self.__cd0[j]*V**2-2*self.__k[j]*self.__W**2/(density_SI(self.h_ref)*self.__S*V**2))/self.__W)*180/numpy.pi
        if self.output == 'SI':
            for i in range(len(self.__k)):
                self.f.write_list('Velocity (m/s)','Climb angle (Deg) condition {}'.format(i+1),numpy.vstack([self.V,gama[:,i]]).T)
        elif self.output == 'imperial':
            for i in range(len(self.__k)):
                self.f.write_list('Velocity (knots)','Climb angle (Deg) condition {}'.format(i+1),numpy.vstack([self.V/0.514444,gama[:,i]]).T)
        
        #Rate-of-climb max
        if self.__propulsion == 'propeller':
            VRC_max = self.VR/numpy.power(3,0.25) #=VP_min, assuming PD not related to V
            RC_max = numpy.zeros_like(self.VR)
            for i in range(len(self.h)):
                for j in range(len(self.__k)):
                    RC_max[i,j] = (self.engine_power_thrust(self.h[i])-2*self.__W*self.VR[i,j]/(numpy.power(3,0.125)*self.__Emax[j]))/self.__W
            #service_ceiling
            service_ceiling = numpy.zeros_like(self.__k)
            for j in range(len(self.__k)):
                for i in range(len(RC_max[:,0])):
                    if RC_max[i,j] <= 100*0.3048/60:  #100 ft/min
                        service_ceiling[j] = self.h[i]
                        break
            # gama_max
            gama_max = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    x0 = self.Vs[i,j]
                    V_gama_max = -1
                    n_iter = 0 
                    while V_gama_max < 0 and n_iter<50:
                        V_gama_max = scipy.optimize.root(self.gama_max_propeller,x0,args=(h,j)).x
                        x0 += 10
                        n_iter += 1
                    gama_max[i,j] = numpy.arcsin(numpy.min([(self.engine_power_thrust(h)/V_gama_max-0.5*density_SI(h)*self.__S*self.__cd0[j]*V_gama_max**2-2*self.__k[j]*self.__W**2/(density_SI(h)*self.__S*V_gama_max**2))/self.__W,1]))*180/numpy.pi
            #Hodograph for h_ref
            V_hod = self.V
            sin_gama_hod = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    sin_gama_hod[i,j] = (self.engine_power_thrust(self.h_ref)/V_hod[i]-0.5*density_SI(self.h_ref)*self.__S*self.__cd0[j]*V_hod[i]**2-2*self.__k[j]*self.__W**2/(density_SI(self.h_ref)*self.__S*V_hod[i]**2))/self.__W
            for i in range(len(sin_gama_hod)):
                for j in range(len(self.__k)):
                    if sin_gama_hod[i,j] < -1:
                        sin_gama_hod[i,j] = -1
                    elif sin_gama_hod[i,j] > 1:
                        sin_gama_hod[i,j] = 1
            RC_hod = sin_gama_hod
            V_hod2plot = []
            RC_hod2plot = []
            for j in range(len(self.__k)):
                V_hod2plot.append([])
                RC_hod2plot.append([])
            for i in range(len(self.V)):
                for j in range(len(self.__k)):
                    if RC_hod[i,j] > 0:
                        V_hod2plot[j].append(V_hod[i]*numpy.sqrt(numpy.max([1-sin_gama_hod[i,j]**2,0])))
                        RC_hod2plot[j].append(RC_hod[i,j])
            plt.figure()
            plt.title('Climb angle at ref altitude and n=1')
            legend = []
            if self.output == 'SI':
                plt.plot(self.V,gama)
                plt.xlabel('Velocity (m/s)')
                plt.ylabel('Climb angle (Deg)')
            elif self.output == 'imperial':
                plt.plot(self.V/0.514444,gama)
                plt.xlabel('Velocity (knots)')
                plt.ylabel('Climb angle (Deg)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(climb_path,'gama_climb_diff_V.png'))
            plt.close()
            plt.figure()
            legend = []
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max rate of climb (m/s) condition {}'.format(i+1),numpy.vstack([self.h,RC_max[:,i]]).T)
                plt.plot(self.h,RC_max)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max rate of climb (m/s)')
            elif self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max rate of climb (ft/s) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,RC_max[:,i]/0.3048]).T)
                plt.plot(self.h/0.3048,RC_max/0.3048)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max rate of climb (ft/s)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(climb_path,'RC_max.png'))
            plt.close()
            plt.figure()
            legend = []
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max climb angle (Deg) condition {}'.format(i+1),numpy.vstack([self.h,gama_max[:,i]]).T)
                plt.plot(self.h,gama_max)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max climb angle (Deg)')
            elif self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max climb angle (Deg) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,gama_max[:,i]]).T)
                plt.plot(self.h/0.3048,gama_max)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max climb angle (Deg)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(climb_path,'gama_max.png'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('{} climb hodograph at ref altitude'.format(self.ac_name))
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    plt.plot(V_hod2plot[i],RC_hod2plot[i])
                plt.xlabel('Velocity (m/s)')
                plt.ylabel('Rate of climb (m/s)')
            elif self.output == 'imperial':
                for i in range(len(self.__k)):
                    plt.plot(numpy.array(V_hod2plot[i])/0.514444,numpy.array(RC_hod2plot[i])/0.3048)
                plt.xlabel('Velocity (knots)')
                plt.ylabel('Rate of climb (ft/s)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(climb_path,'Hodograph.png'))
            plt.close()
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_number('Serv ceiling cond {}'.format(i+1),service_ceiling[i],'m')
            elif self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_number('Serv ceiling cond {}'.format(i+1),service_ceiling[i]/0.3048,'ft')
                
        elif self.__propulsion in ['turbojet','turbofan','ramjet']:
            #jet engine
            # V for max and min Rate-of-climb
            V_RC_min = numpy.zeros_like(self.VR)
            V_RC_max = numpy.zeros_like(self.VR)
            RC_max = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    V_RC_max[i,j] = numpy.sqrt((2*density_SI(h)*self.__S*self.engine_power_thrust(h)+numpy.sqrt((2*density_SI(h)*self.__S*self.engine_power_thrust(h))**2+4*(3*density_SI(h)**2*self.__S**2*self.__cd0[j])*4*self.__k[j]*self.__W**2))/(2*3*density_SI(h)**2*self.__S**2*self.__cd0[j]))
                    V_RC_min[i,j] = numpy.sqrt((2*density_SI(h)*self.__S*self.engine_power_thrust(h)-numpy.sqrt((2*density_SI(h)*self.__S*self.engine_power_thrust(h))**2+4*(3*density_SI(h)**2*self.__S**2*self.__cd0[j])*4*self.__k[j]*self.__W**2))/(2*3*density_SI(h)**2*self.__S**2*self.__cd0[j]))
                   #V_RC_max[i] = numpy.power(self.__k/self.__cd0,0.25)*numpy.sqrt(3*self.engine_power_thrust(h)/(density_SI(h)*self.__S*numpy.sqrt(self.__k*self.__cd0))+2*self.__W/(3*density_SI(h)*self.__S)*numpy.sqrt(self.engine_power_thrust(h)**2/(4*self.__k*self.__cd0*self.__W**2)+3))
                   #V_RC_min[i] = numpy.power(self.__k/self.__cd0,0.25)*numpy.sqrt(3*self.engine_power_thrust(h)/(density_SI(h)*self.__S*numpy.sqrt(self.__k*self.__cd0))-2*self.__W/(3*density_SI(h)*self.__S)*numpy.sqrt(self.engine_power_thrust(h)**2/(4*self.__k*self.__cd0*self.__W**2)+3))
                # Max Rate-of-climb
                    RC_max[i,j] = V_RC_max[i,j]*numpy.min([(self.engine_power_thrust(h)-0.5*density_SI(h)*self.__S*self.__cd0[j]*V_RC_max[i,j]**2-2*self.__k[j]*self.__W**2/(density_SI(h)*self.__S*V_RC_max[i,j]**2))/self.__W,1])
            #service_ceiling
            service_ceiling = numpy.zeros_like(self.__k)
            for j in range(len(self.__k)):
                for i in range(len(RC_max[:,0])):
                    if RC_max[i,j] <= 100*0.3048/60:  #100 ft/min
                        service_ceiling[j] = self.h[i]
                        break
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_number('Serv ceiling cond {}'.format(i+1),service_ceiling[i],'m')
            elif self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_number('Serv ceiling cond {}'.format(i+1),service_ceiling[i]/0.3048,'ft')
            # max climbing angle
            gama_max = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    gama_max[i,j] = numpy.arcsin(self.engine_power_thrust(h)/self.__W-1/self.__Emax[j])*180/numpy.pi
            plt.figure()
            legend = []
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max rate of climb (m/s) condition {}'.format(i+1),numpy.vstack([self.h,RC_max[:,i]]).T)
                plt.plot(self.h,RC_max)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max rate of climb (m/s)')
            elif self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max rate of climb (ft/s) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,RC_max[:,i]/0.3048]).T)
                plt.plot(self.h/0.3048,RC_max/0.3048)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max rate of climb (ft/s)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(climb_path,'RC_max.png'))
            plt.close()
            for i in range(len(self.__k)):
                plt.figure()
                legend = []
                if self.output == 'SI':
                    self.f.write_list('Altitude (m)','V for max rate of climb (m/s) condition {}'.format(i+1),numpy.vstack([self.h,V_RC_max[:,i]]).T)
                    self.f.write_list('Altitude (m)','V for min rate of climb (m/s) condition {}'.format(i+1),numpy.vstack([self.h,V_RC_min[:,i]]).T)
                    plt.plot(self.h,V_RC_max[:,i])
                    plt.plot(self.h,V_RC_min[:,i])
                    plt.xlabel('Altitude (m)')
                    plt.ylabel('Velocity (m/s)')
                elif self.output == 'imperial':
                    self.f.write_list('Altitude (ft)','V for max rate of climb (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,V_RC_max[:,i]/0.514444]).T)
                    self.f.write_list('Altitude (ft)','V for min rate of climb (knots) condition {}'.format(i+1),numpy.vstack([self.h/0.3048,V_RC_min[:,i]/0.514444]).T)
                    plt.plot(self.h/0.3048,V_RC_max[:,i]/0.514444)
                    plt.plot(self.h/0.3048,V_RC_min[:,i]/0.514444)
                    plt.xlabel('Altitude (ft)')
                    plt.ylabel('Velocity (knots)')
                legend.append('V RC max')
                legend.append('V RC min')
                plt.legend(legend,loc='upper left')
                plt.grid()
                plt.savefig(os.path.join(climb_path,'V_RC_max_and_min_cond{}.png'.format(i+1)))
                plt.close()
            plt.figure()
            legend = []
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max climb angle (Deg) condition {}'.format(i+1),numpy.vstack([self.h,gama_max[:,i]]).T)
                plt.plot(self.h,gama_max)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max climb angle (Deg)')
            elif self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max climb angle (Deg) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,gama_max[:,i]]).T)
                plt.plot(self.h/0.3048,gama_max)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max climb angle (Deg)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(climb_path,'gama_max.png'))
            plt.close()

        return
   
    def range_endurance(self,path):
        '''Range and endurance calculations for powered and unpowered flight.'''
        range_path = os.path.join(path,'range')
        if not os.path.isdir(range_path):
            os.mkdir(range_path)
        self.f.header('Range and endurance performance')
        #Unpowered max range
        X_max_u = numpy.zeros_like(self.VR)
        for i in range(len(self.h)):
            for j in range(len(self.__k)):
                X_max_u[i,j] = self.h[i]/(2*numpy.sqrt(self.__k[j]*self.__cd0[j]))
        #Unpowered max endurance, assuming rho constant
        tmax_u = numpy.zeros_like(self.VR)
        for i,h in enumerate(self.h):
            for j in range(len(self.__k)):
                tmax_u[i,j] = 0.25*numpy.sqrt(density_SI(h)*self.__S/(2*self.__W))*numpy.power(27/(self.__cd0[j]*self.__k[j]**3),0.25)*h
        plt.figure()
        legend = []
        plt.title('Unpowered maximum range')
        if self.output == 'SI':
            for i in range(len(self.__k)):
                self.f.write_list('Altitude (m)','Unpowered Max range (m) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),X_max_u[:,i]]).T,scientific=[0,1])
            plt.plot(self.h,X_max_u)
            plt.xlabel('Altitude (m)')
            plt.ylabel('Max range (m)')
        elif self.output == 'imperial':
            for i in range(len(self.__k)):
                self.f.write_list('Altitude (ft)','Unpowered Max range (nm) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,X_max_u[:,i]/(0.514444*3600)]).T)
            plt.plot(self.h/0.3048,X_max_u/(0.514444*3600))
            plt.xlabel('Altitude (ft)')
            plt.ylabel('Max range (nm)')
        for i in range(len(self.__k)):
            legend.append('Condition {}'.format(i+1))
        plt.grid()
        plt.legend(legend,loc='upper left')
        plt.savefig(os.path.join(range_path,'Max_range_for_different_h_unpowered.png'))
        plt.close()
        plt.figure()
        plt.title('Unpowered maximum endurance')
        legend = []
        if self.output == 'SI':
            for i in range(len(self.__k)):
                self.f.write_list('Altitude (m)','Unpowered Max endurance (s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),tmax_u[:,i]]).T,scientific=[0,1])
            plt.plot(self.h,tmax_u)
            plt.xlabel('Altitude (m)')
            plt.ylabel('Max endurance (s)')
        elif self.output == 'imperial':
            for i in range(len(self.__k)):
                self.f.write_list('Altitude (ft)','Unpowered Max endurance (s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,tmax_u[:,i]]).T,scientific=[0,1])
            plt.plot(self.h/0.3048,tmax_u)
            plt.xlabel('Altitude (ft)')
            plt.ylabel('Max endurance (s)')
        for i in range(len(self.__k)):
            legend.append('Condition {}'.format(i+1))
        plt.legend(legend,loc='upper left')
        plt.grid()
        plt.savefig(os.path.join(range_path,'Max_endurance_for_different_h_unpowered.png'))
        plt.close()
        if self.__propulsion == 'propeller':
            #max range
            X_max = numpy.zeros_like(self.VR)
            for i in range(len(self.h)):
                for j in range(len(self.__k)):
                    X_max[i,j] = self.__Emax[j]*self.__eta_p/self.__sfc*numpy.log(self.__W/(self.__W-self.__fuel_W))
            #max endurance for constant h
            tmax_h = numpy.zeros_like(self.VR)
            tmax_V = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    tmax_h[i,j] = 2*self.__eta_p/self.__sfc*numpy.max(self.cl[:,j]**3/self.cd[:,j]**2)*numpy.sqrt(density_SI(h)*self.__S/2)*(1/numpy.sqrt(self.__W-self.__fuel_W)-1/numpy.sqrt(self.__W))
            #max endurance for constant V
                    tmax_V[i,j] = self.__eta_p*self.__Emax[j]*numpy.log(self.__W/(self.__W-self.__fuel_W))/(self.__sfc*self.VR[i,j])
            plt.figure()
            legend = []
            plt.title('Maximum endurance at constant altitude')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max endurance const h(s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),tmax_h[:,i]]).T,scientific=[0,1])
                plt.plot(self.h,tmax_h)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max endurance (s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max endurance const h(s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,tmax_h[:,i]]).T,scientific=[0,1])
                plt.plot(self.h/0.3048,tmax_h)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max endurance (s)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(range_path,'Max_endurance_for_different_h_constant_h.png'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Maximum endurance at constant speed')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max endurance const V(s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),tmax_V[:,i]]).T,scientific=[0,1])
                plt.plot(self.h,tmax_V)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max endurance (s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max endurance cons V(s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,tmax_V[:,i]]).T,scientific=[0,1])
                plt.plot(self.h/0.3048,tmax_V)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max endurance (s)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(range_path,'Max_endurance_for_different_h_constant_V.png'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Maximum range')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max range(m) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),X_max[:,i]]).T,scientific=[0,1])
                plt.plot(self.h,X_max)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max range (m)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max range(nm) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,X_max[:,i]/(0.514444*3600)]).T)
                plt.plot(self.h/0.3048,X_max/(0.514444*3600))
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max range (nm)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(range_path,'Max_range_for_different_h_powered.png'))
            plt.close()
        elif self.__propulsion in ['turbojet','turbofan','ramjet']:
            #max range for constant h (V=numpy.power(3,0.25)*VR)
            X_max_h = numpy.zeros_like(self.VR)
            X_max_V = numpy.zeros_like(self.VR)
            tmax = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    X_max_h[i,j] = numpy.sqrt(self.__cl_star[j]/numpy.sqrt(3))/(self.__cd0[j]+self.__k[j]*self.__cl_star[j]**2/3)*(2/self.__sfc)*numpy.sqrt(2/(density_SI(h)*self.__S))*(-numpy.sqrt(self.__W-self.__fuel_W)+numpy.sqrt(self.__W))
            #max range for constant V (V=VR)
                    X_max_V[i,j] = self.__Emax[j]*self.Vcruise*numpy.log(self.__W/(self.__W-self.__fuel_W))/self.__sfc
            #max endurance
                    tmax[i,j] = self.__Emax[j]*numpy.log(self.__W/(self.__W-self.__fuel_W))/self.__sfc
            plt.figure()
            legend = []
            plt.title('Maximum endurance')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max endurance(s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),tmax[:,i]]).T,scientific=[0,1])
                plt.plot(self.h,tmax)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max endurance (s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max endurance(s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,tmax[:,i]]).T,scientific=[0,1])
                plt.plot(self.h/0.3048,tmax)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max endurance (s)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend,loc='upper left')
            plt.grid()
            plt.savefig(os.path.join(range_path,'Max_endurance_for_different_h_powered.png'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Maximum range at constant speed')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max range const V(m) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),X_max_V[:,i]]).T,scientific=[0,1])
                plt.plot(self.h,X_max_V)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max range (m)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max range const V(nm) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,X_max_V[:,i]/(0.514444*3600)]).T)
                plt.plot(self.h/0.3048,X_max_V/(0.514444*3600))
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max range (nm)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend,loc='upper left')
            plt.grid()
            plt.savefig(os.path.join(range_path,'Max_range_for_different_h_constant_V.png'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Maximum range at constant altitude')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Max range const h(m) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),X_max_h[:,i]]).T,scientific=[0,1])
                plt.plot(self.h,X_max_h)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Max range (m)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Max range const h(nm) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,X_max_h[:,i]/(0.514444*3600)]).T)
                plt.plot(self.h/0.3048,X_max_h/(0.514444*3600))
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Max range (nm)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend,loc='upper left')
            plt.grid()
            plt.savefig(os.path.join(range_path,'Max_range_for_different_h_constant_h.png'))
            plt.close()
        return

    def omega_max_propeller(self,V,h,j):
        '''Calculates V for different altitudes that gives the max turn rate.'''
        n2 = (self.__S/self.__W)**2*(self.engine_power_thrust(h)*density_SI(h)*V/(2*self.__S*self.__k[j])-density_SI(h)**2*self.__cd0[j]*V**4/(4*self.__k[j]))
        n_clmax = 0.5*density_SI(h)*V**2*self.__clmax[j]*self.__S/self.__W
        if n2 >= self.__nlim**2 or n2 >= n_clmax**2:
            n2 = min([self.__nlim**2,n_clmax**2])
        eq = (self.__S/self.__W)**2*(self.engine_power_thrust(h)*density_SI(h)/(2*self.__k[j]*self.__S)-density_SI(h)**2*self.__cd0[j]*V**3/(4*self.__k[j]))-2*(n2-1)/V
        return eq

    def R_min_propeller(self,V,h,j):
        '''Calculates V for different altitudes that gives the min R.'''
        n2 = (self.__S/self.__W)**2*(self.engine_power_thrust(h)*density_SI(h)*V/(2*self.__S*self.__k[j])-density_SI(h)**2*self.__cd0[j]*V**4/(4*self.__k[j]))
        n_clmax = 0.5*density_SI(h)*V**2*self.__clmax[j]*self.__S/self.__W
        if n2 >= self.__nlim**2 or n2 >= n_clmax**2:
            n2 = min([self.__nlim**2,n_clmax**2])
        eq = (self.__S/self.__W)**2*(self.engine_power_thrust(h)*density_SI(h)/(2*self.__k[j]*self.__S)-density_SI(h)**2*self.__cd0[j]*V**3/(4*self.__k[j]))-4*(n2-1)/V
        return eq

    def R_min_jet(self,u,i,h,z,j):
        '''Calculates V for different altitudes that gives the min R.'''
        n2 = u**2*(2*z-u**2)
        n_clmax = 0.5*density_SI(h)*u*self.VR[i,j]**2*self.__clmax[j]*self.__S/self.__W
        if n2 >= self.__nlim**2 or n2 >= n_clmax**2:
            n2 = min([self.__nlim**2,n_clmax**2])
        R = u**2*self.VR[i,j]**2/(9.81*numpy.sqrt(n2-1))
        return R
        

    def turn_performance(self,path):
        '''Turning calculations for powered and unpowered flight.'''
        turn_path = os.path.join(path,'turn')
        if not os.path.isdir(turn_path):
            os.mkdir(turn_path)
        self.f.header('Turn performance')

      ###instantaneous turn for specified parameters.
      ##beta = numpy.linspace(-numpy.pi/2,numpy.pi/2,1000) #slip angle
      ##phi = numpy.linspace(0,3*numpy.pi/8,1000) #roll angle
      ###radius
      ##R = numpy.zeros((len(phi),len(beta)))
      ##for i in range(len(phi)):
      ##    for j in range(len(beta)):
      ##        R[i,j] = self.VA**2/(9.81*(numpy.tan(phi[i])*self.engine_power_thrust(self.h_ref)*numpy.sin(beta[j])/self.__W))
      ##omega = self.VA/R
      ### time to complete a 360 deg turn.
      ##t = 2*numpy.pi/omega

        #unpowered turn
         

        # sustained turn
        if self.__propulsion == 'propeller':
            #max load factor:
            V_nmax = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    V_nmax[i,j] = numpy.power(self.engine_power_thrust(h)/(2*density_SI(h)*self.__S*self.__cd0[j]),1.0/3.0)
            nmax = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    nmax[i,j] = 0.6874*numpy.power(self.engine_power_thrust(h)**2*density_SI(h)*self.__S*self.__Emax[j]/(self.__k[j]*self.__W**3),1.0/3.0)
            phi_max = numpy.arccos(1/nmax)*180/numpy.pi
            
            #max omega:
            sol = numpy.zeros_like(self.VR)
            n_om = numpy.zeros_like(self.VR)
            for j in range(len(self.__k)):
                for i,h in enumerate(self.h):
                    if i == 0:
                        sol[i,j] = scipy.optimize.ridder(self.omega_max_propeller,0,self.V[-1],args=(h,j))
                        n_om[i,j] = (self.__S/self.__W)*numpy.sqrt(self.engine_power_thrust(h)*density_SI(h)*sol[i,j]/(2*self.__S*self.__k[j])-density_SI(h)**2*self.__cd0[j]*sol[i,j]**4/(4*self.__k[j]))
                    else:
                        x0 = 5
                        niter = 0
                        while numpy.absolute(sol[i,j]-sol[i-1,j])/sol[i-1,j] > 0.05 and niter<15:
                            sol[i,j] = scipy.optimize.ridder(self.omega_max_propeller,0,self.V[-1],args=(h,j))
                            n_om[i,j] = (self.__S/self.__W)*numpy.sqrt(self.engine_power_thrust(h)*density_SI(h)*sol[i,j]/(2*self.__S*self.__k[j])-density_SI(h)**2*self.__cd0[j]*sol[i,j]**4/(4*self.__k[j]))
                            x0+=5
                            niter+=1
            V_om = sol
            omega_max = 9.81*numpy.sqrt(n_om**2-1)/sol

            #R min
            sol = numpy.zeros_like(self.VR)
            n_R = numpy.zeros_like(self.VR)
            for j in range(len(self.__k)):
                for i,h in enumerate(self.h):
                    if i == 0:
                        sol[i,j] = scipy.optimize.ridder(self.R_min_propeller,0,self.V[-1],args=(h,j))
                    else:
                        x0 = 5 
                        niter = 0
                        while numpy.absolute(sol[i,j]-sol[i-1,j])/sol[i-1,j] > 0.05 and niter<15:
                            sol[i,j] = scipy.optimize.ridder(self.R_min_propeller,0,self.V[-1],args=(h,j))
                            x0+=5
                            niter+=1
                    n_R[i,j] = (self.__S/self.__W)*numpy.sqrt(self.engine_power_thrust(h)*density_SI(h)*sol[i,j]/(2*self.__S*self.__k[j])-density_SI(h)**2*self.__cd0[j]*sol[i,j]**4/(4*self.__k[j]))
            V_R = sol
            R_min = []
            h_rmin = []
            for j in range(len(self.__k)):
                R_min.append([])
                h_rmin.append([])
            for j in range(len(self.__k)):
                for i,h in enumerate(self.h):
                    if i == 0:
                        R_min[j].append(V_R[0,j]**2/(9.81*numpy.sqrt(n_R[0,j]**2-1)))
                        h_rmin[j].append(h)
                    elif V_R[i,j]**2/(9.81*numpy.sqrt(n_R[i,j]**2-1)) < 20*R_min[j][0]:
                        R_min[j].append(V_R[i,j]**2/(9.81*numpy.sqrt(n_R[i,j]**2-1)))
                        h_rmin[j].append(h)

            #plots
            plt.figure()
            legend = []
            plt.title('Velocity for maximum load factor')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','V n max(m/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),V_nmax[:,i]]).T)
                plt.plot(self.h,V_nmax)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Velocity (m/s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','V n max(knots) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,V_nmax[:,i]/(0.514444)]).T)
                plt.plot(self.h/0.3048,V_nmax/(0.514444))
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Velocity (knots)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Velocity for maximum turn load factor for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Maximum load factor')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','n max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),nmax[:,i]]).T)
                plt.plot(self.h,nmax)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Load factor')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','n max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,nmax[:,i]]).T)
                plt.plot(self.h/0.3048,nmax)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Load factor')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Maximum turn load factor for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Maximum roll angle')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Phi max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),phi_max[:,i]]).T)
                plt.plot(self.h,phi_max)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Phi (Deg)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Phi max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,phi_max[:,i]]).T)
                plt.plot(self.h/0.3048,phi_max)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Phi (Deg)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Maximum roll angle for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Maximum turn ratio')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Omega max(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),omega_max[:,i]]).T)
                plt.plot(self.h,omega_max)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Omega (Rad/s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Omega max(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,omega_max[:,i]]).T)
                plt.plot(self.h/0.3048,omega_max)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Omega (Rad/s)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Maximum turn ratio for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Velocity for maximum turn ratio')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','V omega max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),V_om[:,i]]).T)
                plt.plot(self.h,V_om)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Velocity (m/s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','V omega max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,V_om[:,i]/0.514444]).T)
                plt.plot(self.h/0.3048,V_om/0.514444)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Velocity (knots)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Velocity for maximum omega for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Load factor for maximum turn ratio')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','n omega max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),n_om[:,i]]).T)
                plt.plot(self.h,n_om)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Load factor')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','n omega max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,n_om[:,i]]).T)
                plt.plot(self.h/0.3048,n_om)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Load factor')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Load factor for maximum omega for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Minimum turn radius')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','R min condition {}'.format(i+1),numpy.vstack([numpy.array(h_rmin[i]),R_min[i]]).T)
                    plt.plot(h_rmin[i],R_min[i])
                plt.xlabel('Altitude (m)')
                plt.ylabel('Radius (m)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','R min condition {}'.format(i+1),numpy.vstack([numpy.array(h_rmin[i])/0.3048,numpy.array(R_min[i])/0.3048]).T)
                    plt.plot(numpy.array(h_rmin[i])/0.3048,numpy.array(R_min[i])/0.3048)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Radius (ft)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Minimum turn radius for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Velocity for minimum turn radius')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','V R min condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),V_R[:,i]]).T)
                plt.plot(self.h,V_R)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Velocity (m/s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','V R min condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,V_R[:,i]/0.514444]).T)
                plt.plot(self.h/0.3048,V_R/0.514444)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Velocity (knots)')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Velocity for minimum R for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Load factor for minimum turn radius')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','n R min condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),n_R[:,i]]).T)
                plt.plot(self.h,n_R)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Load factor')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','n R min condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,n_R[:,i]]).T)
                plt.plot(self.h/0.3048,n_R)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Load factor')
            for i in range(len(self.__k)):
                legend.append('Condition {}'.format(i+1))
            plt.legend(legend)
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Load factor for minimum R for different h.'))
            plt.close()
        elif self.__propulsion in ['turbojet','turbofan','ramjet']:
            #adimensional parameters
            #u=V/VR
            z=numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    z[i,j] = self.engine_power_thrust(h)*self.__Emax[j]/self.__W
            #max load factor:
            n_max = numpy.ones_like(self.VR)
            V_n = numpy.ones_like(self.VR)
            for iter in range(20):
                V_n = numpy.sqrt(n_max)*self.VR
                for i,h in enumerate(self.h):
                    for j in range(len(self.__k)):
                        n_max[i,j] = numpy.min([z[i,j],0.5*density_SI(h)*V_n[i,j]**2*self.__clmax[j]*self.__S/self.__W,self.__nlim])
            R_n = V_n**2/(9.81*numpy.sqrt(n_max**2-1))
            omega_n = V_n/R_n
            #max omega:
            #V=VR
            n_w = numpy.zeros_like(self.VR)
            cl_w = numpy.zeros_like(self.VR)
            for i,h in enumerate(self.h):
                for j in range(len(self.__k)):
                    n_w[i,j] = numpy.min([numpy.sqrt(2*z-1)[i,j],0.5*density_SI(h)*self.VR[i,j]**2*self.__clmax[j]*self.__S/self.__W,self.__nlim])
                    cl_w[i,j] = n_w[i,j]*self.__cl_star[j]
            omega_max = 9.81*numpy.sqrt(n_w**2-1)/self.VR
            R_w = self.VR/omega_max
            #R min
            x0 = [1] #u
            V_R = numpy.zeros_like(self.VR)
            n_R = numpy.zeros_like(self.VR)
            for j in range(len(self.__k)):
                for i,h in enumerate(self.h):
                    u = scipy.optimize.fminbound(self.R_min_jet,0,self.V[-1]/self.VR[i,j],args=(i,h,z[i,j],j))
                    V_R[i,j] = u*self.VR[i,j]    
                    n_R[i,j] = numpy.min([u*numpy.sqrt(2*z[i,j]-u**2),0.5*density_SI(h)*u*self.VR[i,j]**2*self.__clmax[j]*self.__S/self.__W,self.__nlim])
                    x0[0] = u
            R_min = V_R**2/(9.81*numpy.sqrt(n_R**2-1))
            omega_R = V_R/R_min
            #plots
            plt.figure()
            legend = []
            plt.title('Relevant velocities for turn performance')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','V n max (m/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),V_n[:,i]]).T)
                    self.f.write_list('Altitude (m)','V omega max (m/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),self.VR[:,i]]).T)
                    self.f.write_list('Altitude (m)','V R min (m/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),V_R[:,i]]).T)
                plt.plot(self.h,V_n)
                plt.plot(self.h,self.VR)
                plt.plot(self.h,V_R)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Velocity (m/s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','V n max (knots) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,V_n[:,i]/0.514444]).T)
                    self.f.write_list('Altitude (ft)','V omega max (knots) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,self.VR[:,i]/0.514444]).T)
                    self.f.write_list('Altitude (ft)','V R min (knots) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,V_R[:,i]/0.514444]).T)
                plt.plot(self.h/0.3048,V_n/0.514444)
                plt.plot(self.h/0.3048,self.VR/0.514444)
                plt.plot(self.h/0.3048,V_R/0.514444)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Velocity (knots)')
            for i in range(len(self.__k)):
                legend.append('V n max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('V omega max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('V R min cond {}'.format(i+1))
            plt.legend(legend,loc='upper left', prop={'size':10})
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Turn velocity plots for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Relevant load factors for turn performance')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','n max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),n_max[:,i]]).T)
                    self.f.write_list('Altitude (m)','n omega max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),n_w[:,i]]).T)
                    self.f.write_list('Altitude (m)','n R min condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),n_R[:,i]]).T)
                plt.plot(self.h,n_max)
                plt.plot(self.h,n_w)
                plt.plot(self.h,n_R)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Load factor')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','n max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,n_max[:,i]]).T)
                    self.f.write_list('Altitude (ft)','n omega max condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,n_w[:,i]]).T)
                    self.f.write_list('Altitude (ft)','n R min condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,n_R[:,i]]).T)
                plt.plot(self.h/0.3048,n_max)
                plt.plot(self.h/0.3048,n_w)
                plt.plot(self.h/0.3048,n_R)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Load factor')
            for i in range(len(self.__k)):
                legend.append('n max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('n omega max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('n R min cond {}'.format(i+1))
            plt.legend(legend,loc='lower left', prop={'size':10})
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Turn load factor plots for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Turn radius for different conditions')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','R n max (m) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),R_n[:,i]]).T)
                    self.f.write_list('Altitude (m)','R omega max (m) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),R_w[:,i]]).T)
                    self.f.write_list('Altitude (m)','R min (m) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),R_min[:,i]]).T)
                plt.plot(self.h,R_n)
                plt.plot(self.h,R_w)
                plt.plot(self.h,R_min)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Turn radius (m)')
                plt.ylim([0.0,10*R_n[0,0]])
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','R n max (ft) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,R_n[:,i]/0.3048]).T)
                    self.f.write_list('Altitude (ft)','R omega max (ft) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,R_w[:,i]/0.3048]).T)
                    self.f.write_list('Altitude (ft)','R min (ft) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,R_min[:,i]/0.3048]).T)
                plt.plot(self.h/0.3048,R_n/0.3048)
                plt.plot(self.h/0.3048,R_w/0.3048)
                plt.plot(self.h/0.3048,R_min/0.3048)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Turn radius (ft)')
                plt.ylim([0.0,10*R_n[0,0]/0.3048])
            for i in range(len(self.__k)):
                legend.append('R n max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('R omega max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('R min cond {}'.format(i+1))
            plt.legend(legend,loc='upper left', prop={'size':10})
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Turn radius plots for different h.'))
            plt.close()
            plt.figure()
            legend = []
            plt.title('Turn ratio for different conditions')
            if self.output == 'SI':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (m)','Omega n max(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),omega_n[:,i]]).T)
                    self.f.write_list('Altitude (m)','Omega max(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),omega_max[:,i]]).T)
                    self.f.write_list('Altitude (m)','Omega R min(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h),omega_R[:,i]]).T)
                plt.plot(self.h,omega_n)
                plt.plot(self.h,omega_max)
                plt.plot(self.h,omega_R)
                plt.xlabel('Altitude (m)')
                plt.ylabel('Omega (Rad/s)')
            if self.output == 'imperial':
                for i in range(len(self.__k)):
                    self.f.write_list('Altitude (ft)','Omega n max(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,omega_n[:,i]]).T)
                    self.f.write_list('Altitude (ft)','Omega max(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,omega_max[:,i]]).T)
                    self.f.write_list('Altitude (ft)','Omega R min(rad/s) condition {}'.format(i+1),numpy.vstack([numpy.array(self.h)/0.3048,omega_R[:,i]]).T)
                plt.plot(self.h/0.3048,omega_n)
                plt.plot(self.h/0.3048,omega_max)
                plt.plot(self.h/0.3048,omega_R)
                plt.xlabel('Altitude (ft)')
                plt.ylabel('Omega(Rad/s)')
            for i in range(len(self.__k)):
                legend.append('Omega n max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('Omega max cond {}'.format(i+1))
            for i in range(len(self.__k)):
                legend.append('Omega R min cond {}'.format(i+1))
            plt.legend(legend, prop={'size':10})
            plt.grid()
            plt.savefig(os.path.join(turn_path,'Turn ratio plots for different h.'))
            plt.close()
            
        return

    def take_off_landing(self,path):
        '''Takeoff and landing calculations.'''
        toff_path = os.path.join(path,'take_off_landing')
        if not os.path.isdir(toff_path):
            os.mkdir(toff_path)
        self.f.header('Take-off and landing')
        #Takeoff
        self.h_toff = numpy.linspace(0,4400,50)
        #rolling groung friction coefficient(mi) for concrete, short grass, long grass and soft ground
        mig = [0.025,0.05,0.1,0.2]
        V1 = numpy.zeros((len(self.h_toff),len(self.__k)))
        F0 = numpy.zeros((len(self.h_toff),len(self.__k),len(mig)))
        F1 = numpy.zeros((len(self.h_toff),len(self.__k),len(mig)))
        s1 = numpy.zeros((len(self.h_toff),len(self.__k),len(mig)))
        for i,h in enumerate(self.h_toff):
            for j in range(len(self.__k)):
                for k,mi in enumerate(mig):
                    V1[i,j] = 1.1*numpy.sqrt(self.__W/(0.5*density_SI(h)*self.__S*self.__clmax[j])) #FAR23, FAR25, militar
                    if self.__propulsion == 'propeller':
                        F0[i,j,k] = numpy.max([self.engine_power_thrust(h)/V1[i,j] - mi*self.__W - self.theta*self.__W,0])
                        F1[i,j,k] = numpy.max([self.engine_power_thrust(h)/V1[i,j] - 0.5*density_SI(h)*V1[i,j]**2*self.__S*(self.__cd0[j]+self.__k[j]*(mi/(2*self.__k[j]))**2),0])
                    else:
                        F0[i,j,k] = numpy.max([self.engine_power_thrust(h) - mi*self.__W - self.theta*self.__W,0])
                        F1[i,j,k] = numpy.max([self.engine_power_thrust(h) - 0.5*density_SI(h)*V1[i,j]**2*self.__S*(self.__cd0[j]+self.__k[j]*(mi/(2*self.__k[j]))**2),0])
        #ground distance (for s1 min, cl = mi/2k)
                    s1[i,j,k] = self.__W/(2*9.81)*(V1[i,j]**2/(F0[i,j,k]-F1[i,j,k]))*numpy.log(F0[i,j,k]/F1[i,j,k])
        t1 = numpy.zeros_like(s1)
        for i,h in enumerate(self.h_toff):
            for j in range(len(self.__k)):
                for k,mi in enumerate(mig):
                    if F1[i,j,k]>F0[i,j,k]:
                        t1[i,j,k] = (self.__W*V1[i,j])/(9.81*numpy.sqrt(F0[i,j,k]*(F1[i,j,k]-F0[i,j,k])/V1[i,j]**2))*numpy.arctan((F1[i,j,k]-F0[i,j,k])/(F0[i,j,k]*V1[i,j]**2))
                    else:
                        t1[i,j,k] = (self.__W*V1[i,j])/(2*9.81*numpy.sqrt(F0[i,j,k]*(F1[i,j,k]-F0[i,j,k])/V1[i,j]**2))*numpy.log((numpy.sqrt(F0[i,j,k])+V1[i,j]*numpy.sqrt(numpy.absolute((F1[i,j,k]-F0[i,j,k])/V1[i,j]**2)))/(numpy.sqrt(F0[i,j,k])-V1[i,j]*numpy.sqrt(numpy.absolute((F1[i,j,k]-F0[i,j,k])/V1[i,j]**2))))
        h_screen = 35*0.3048 #obstacle hight = 35ft
        sin_gama = numpy.zeros_like(V1)
        for i,h in enumerate(self.h_toff):
            for j in range(len(self.__k)):
                sin_gama[i] = (self.engine_power_thrust(h)-0.5*density_SI(h)*V1[i,j]**2*self.__S*(self.__cd0[j]+self.__k[j]*(self.__W/(0.5*density_SI(h)*V1[i,j]**2*self.__S))**2))/self.__W #n=1
        gama = numpy.arcsin(sin_gama)
        #climb distance
        s2= h_screen/numpy.tan(gama)
        t2= s2/(V1*numpy.cos(gama))
        #plots
        mi_name=['concrete','short grass','long grass','soft ground']
        color_style1 = ['-b','-k','-r','-g']
        line_style1 = ['x','o','+','v']
        line_style3 = ['o','+','v','x']
        for i in range(len(self.__k)):
            plt.figure()
            plt.title('Take-off distance at different altitudes.')
            legend=[]
            if self.output == 'SI':
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff,s1[:,i,j],color_style1[j]+line_style1[j])
                    legend.append('Ground distance ({})'.format(mi_name[j]))
                    self.f.write_list('Altitude (m)','Ground Toff distance(m) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff,s1[:,i,j]]).T)
                plt.plot(self.h_toff,s2[:,i])
                self.f.write_list('Altitude (m)','Climb Toff distance(m) cond {}'.format(i+1),numpy.vstack([self.h_toff,s2[:,i]]).T)
                legend.append('Climb distance')
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff,s1[:,i,j]+s2[:,i],color_style1[j]+line_style3[j])
                    self.f.write_list('Altitude (m)','Total Toff distance(m) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff,s1[:,i,j]+s2[:,i]]).T)
                    legend.append('Total distance ({})'.format(mi_name[j]))
                plt.xlabel('Track altitude (m)')
                plt.ylabel('Distance (m)')
            elif self.output == 'imperial':
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff/0.3048,s1[:,i,j]/0.3048,color_style1[j]+line_style1[j])
                    legend.append('Ground distance ({})'.format(mi_name[j]))
                    self.f.write_list('Altitude (ft)','Ground Toff distance(ft) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff/0.3048,s1[:,i,j]/0.3048]).T)
                plt.plot(self.h_toff/0.3048,s2[:,i]/0.3048)
                self.f.write_list('Altitude (ft)','Climb Toff distance(ft) cond {}'.format(i+1),numpy.vstack([self.h_toff/0.3048,s2[:,i]/0.3048]).T)
                legend.append('Climb distance')
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff/0.3048,s1[:,i,j]/0.3048+s2[:,i]/0.3048,color_style1[j]+line_style3[j])
                    self.f.write_list('Altitude (ft)','Total Toff distance(ft) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff/0.3048,s1[:,i,j]/0.3048+s2[:,i]/0.3048]).T)
                    legend.append('Total distance ({})'.format(mi_name[j]))
                plt.xlabel('Track altitude (ft)')
                plt.ylabel('Distance (ft)')
            plt.legend(legend,loc='upper left',fontsize='x-small')
            plt.grid()
            plt.savefig(os.path.join(toff_path,'Takeoff distance for different tracks and h cond{}.'.format(i+1)))
        plt.close()
        for i in range(len(self.__k)):
            plt.figure()
            plt.title('Take-off time at different altitudes.')
            legend=[]
            if self.output == 'SI':
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff,t1[:,i,j],color_style1[j]+line_style1[j])
                    legend.append('Ground time ({})'.format(mi_name[j]))
                    self.f.write_list('Altitude (m)','Ground Toff time(s) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff,t1[:,i,j]]).T)
                plt.plot(self.h_toff,t2[:,i])
                self.f.write_list('Altitude (m)','Climb Toff time(s) cond {}'.format(i+1),numpy.vstack([self.h_toff,t2[:,i]]).T)
                legend.append('Climb time')
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff,t1[:,i,j]+t2[:,i],color_style1[j]+line_style3[j])
                    self.f.write_list('Altitude (m)','Total Toff time(s) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff,t1[:,i,j]+t2[:,i]]).T)
                    legend.append('Total time ({})'.format(mi_name[j]))
                plt.xlabel('Track altitude (m)')
                plt.ylabel('Time (s)')
            elif self.output == 'imperial':
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff/0.3048,t1[:,i,j],color_style[j]+line_style1[j])
                    legend.append('Ground time ({})'.format(mi_name[j]))
                    self.f.write_list('Altitude (ft)','Ground Toff time(s) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff/0.3048,t1[:,i,j]]).T)
                plt.plot(self.h_toff/0.3048,t2[:,i])
                self.f.write_list('Altitude (ft)','Climb Toff time(s) cond {}'.format(i+1),numpy.vstack([self.h_toff/0.3048,t2[:,i]]).T)
                legend.append('Climb time')
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff/0.3048,t1[:,i,j]+t2[:,i],color_style[j]+line_style3[j])
                    self.f.write_list('Altitude (ft)','Total Toff time(s) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff/0.3048,t1[:,i,j]+t2[:,i]]).T)
                    legend.append('Total time ({})'.format(mi_name[j]))
                plt.xlabel('Track altitude (ft)')
                plt.ylabel('Time (s)')
            plt.legend(legend,loc='upper left',fontsize='x-small')
            plt.grid()
            plt.savefig(os.path.join(toff_path,'Takeoff time for different tracks and h cond{}.'.format(i+1)))
        plt.close()
       
        ##Landing
        mig = [0.025,0.05,0.1,0.2]
       #mig = [0.175,0.2,0.25,0.35]
        VA = numpy.zeros_like(V1) #approach speed (FAR)
        for i,h in enumerate(self.h_toff):
            for j,k in enumerate(self.__k):
                VA[i,j] = 1.3*numpy.sqrt(self.__W/(0.5*density_SI(h)*self.__S*self.__clmax[k])) #FAR23, FAR25, militar
        gama = 3*numpy.pi/180 #approach angle (assumed 3 Deg)
        h_obst = 50*0.3048 #Obstacle hight
        #Approach
        s1_l = h_obst/numpy.tan(gama)*numpy.ones_like(V1)
        t1_l = s1_l/(VA*numpy.cos(gama))
        #Flare
        R_l = VA**2/(0.69*9.81)
        s2_l = VA**2*gama/(0.69*9.81)
        #Ground run
        s3_l = numpy.zeros_like(s1)
        for i,h in enumerate(self.h_toff):
            for j in range(len(self.__k)):
                for k,mi in enumerate(mig):
                    if self.__propulsion == 'propeller':
                        F0[i,j,k] = numpy.max([0.1*self.engine_power_thrust(h)/VA[i,j] + mi*self.__W,0])
                        F1[i,j,k] = numpy.max([0.1*self.engine_power_thrust(h)/VA[i,j] + 0.5*density_SI(h)*VA[i,j]**2*self.__S*(self.__cd0[j]+self.__k[j]*(mi/(2*self.__k[j]))**2),0])
                    else:
                        F0[i,j,k] = numpy.max([0.1*self.engine_power_thrust(h) + mi*self.__W,0])
                        F1[i,j,k] = numpy.max([0.1*self.engine_power_thrust(h) + 0.5*density_SI(h)*VA[i,j]**2*self.__S*(self.__cd0[j]+self.__k[j]*(mi/(2*self.__k[j]))**2),0])
        #considering residual Thrust = 0.1*max thrust
                    s3_l[i,j,k] = self.__W/(2*9.81)*(VA[i,j]**2/(F1[i,j,k]-F0[i,j,k]))*numpy.log(F1[i,j,k]/F0[i,j,k])
        #plots
        for i in range(len(self.__k)):
            plt.figure()
            plt.title('Landing distance at different altitudes.')
            legend=[]
            if self.output == 'SI':
                plt.plot(self.h_toff,s1_l[:,i])
                self.f.write_list('Altitude (m)','Approach distance(m) cond {}'.format(i+1),numpy.vstack([self.h_toff,s1_l[:,i]]).T)
                legend.append('Approach distance')
                plt.plot(self.h_toff,s2_l[:,i])
                self.f.write_list('Altitude (m)','Flare distance(m) cond {}'.format(i+1),numpy.vstack([self.h_toff,s2_l[:,i]]).T)
                legend.append('Flare distance')
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff,s3_l[:,i,j],color_style1[j]+line_style3[j])
                    self.f.write_list('Altitude (m)','Ground run distance(m) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff,s3_l[:,i,j]]).T)
                    legend.append('Ground run distance ({})'.format(mi_name[j]))
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff,s1_l[:,i]+s2_l[:,i]+s3_l[:,i,j],color_style1[j]+line_style3[j])
                    self.f.write_list('Altitude (m)','Total landing distance(m) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff,s1_l[:,i]+s2_l[:,i]+s3_l[:,i,j]]).T)
                    legend.append('Total distance ({})'.format(mi_name[j]))
                plt.xlabel('Track altitude (m)')
                plt.ylabel('Distance (m)')
            elif self.output == 'imperial':
                plt.plot(self.h_toff/0.3048,s1_l[:,i]/0.3048)
                self.f.write_list('Altitude (ft)','Approach distance(ft) cond {}'.format(i+1),numpy.vstack([self.h_toff/0.3048,s1_l[:,i]/0.3048]).T)
                legend.append('Approach distance')
                plt.plot(self.h_toff/0.3048,s2_l[:,i]/0.3048)
                self.f.write_list('Altitude (ft)','Flare distance(ft) cond {}'.format(i+1),numpy.vstack([self.h_toff/0.3048,s2_l[:,i]/0.3048]).T)
                legend.append('Flare distance')
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff/0.3048,s3_l[:,i,j]/0.3048,color_style1[j]+line_style3[j])
                    self.f.write_list('Altitude (ft)','Ground run distance(ft) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff/0.3048,s3_l[:,i,j]/0.3048]).T)
                    legend.append('Ground run distance ({})'.format(mi_name[j]))
                for j,mi in enumerate(mig):
                    plt.plot(self.h_toff/0.3048,s1_l[:,i]/0.3048+s2_l[:,i]/0.3048+s3_l[:,i,j]/0.3048,color_style1[j]+line_style3[j])
                    self.f.write_list('Altitude (ft)','Total landing distance(ft) {} cond {}'.format(mi_name[j],i+1),numpy.vstack([self.h_toff/0.3048,s1_l[:,i]/0.3048+s2_l[:,i]/0.3048+s3_l[:,i,j]/0.3048]).T)
                    legend.append('Total distance ({})'.format(mi_name[j]))
                plt.xlabel('Track altitude (ft)')
                plt.ylabel('Distance (ft)')
            plt.legend(legend,loc='upper left',fontsize='xx-small')
            plt.grid()
            plt.savefig(os.path.join(toff_path,'Landing distance for different tracks and h cond{}.'.format(i+1)))
        plt.close()
        plt.figure()
        plt.title('Landing approach time at different altitudes.')
        legend=[]
        if self.output == 'SI':
            plt.plot(self.h_toff,t1_l)
            for i in range(len(self.__k)):
                self.f.write_list('Altitude (m)','Approach time(s) cond {}'.format(i+1),numpy.vstack([self.h_toff,t1_l[:,i]]).T)
        elif self.output == 'imperial':
            plt.plot(self.h_toff/0.3048,t1_l)
            for i in range(len(self.__k)):
                self.f.write_list('Altitude (ft)','Approach time(s) cond {}'.format(i+1),numpy.vstack([self.h_toff/0.3048,t1_l[:,i]]).T)
        for i in range(len(self.__k)):
            legend.append('Condition {}'.format(i+1))
        plt.grid()
        plt.legend(legend)
        plt.savefig(os.path.join(toff_path,'Landing time.'))
        plt.close()
        
        return
########################################################################

#Main

def performance_calculations(dat_path,out_path):
    '''Performs basic performance calculations. Every variable is converted to SI and then, after all the calculations, converted to the desired unit.'''
    data_dict = read_dat(dat_path)
    #Reading the input file data:
    output = data_dict['output'][0]
    ac_name = ' '.join(x for x in data_dict['ac_name'])
    out_path = os.path.join(out_path,ac_name)
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    #Opens the output log file to save the data:
    outfile = os.path.join(out_path,ac_name+'.log')
    description = 'File containing the relevant data for performance analysis.'
    f = File(outfile,description)
    f.header('Input data')
    f.write_txt('Aircraft name',ac_name)
    f.write_txt('Output units',output)
    var = Variable(data_dict['W'])
    f.write_number('Aircraft Weight',var.value,var.unit,scientific=1)
    var.convert2SI()
    W = var.value
    var = Variable(data_dict['S'])
    f.write_number('Wing Area',var.value,var.unit)
    var.convert2SI()
    S = var.value
    var = Variable(data_dict['df'])
    f.write_number('Fuselage D at wing r',var.value,var.unit)
    var.convert2SI()
    df = var.value
    var = Variable(data_dict['b'])
    f.write_number('Wing Span',var.value,var.unit)
    var.convert2SI()
    b = var.value
    var = Variable(data_dict['c'])
    f.write_number('Mean Aero Chord',var.value,var.unit)
    var.convert2SI()
    c = var.value
    sweep = float(data_dict['sweep'][0])
    f.write_number('Wing 1/4c Sweep',sweep,'Deg')
    theta = float(data_dict['theta'][0])
    f.write_number('Takeoff AoA',theta,'Deg')
    clmax = []
    for i,cl in enumerate(data_dict['cl_max']):
        clmax.append(float(cl))
        f.write_number('Cl max condition {}'.format(i+1),cl,' ')
    nlim = float(data_dict['nlim'][0])
    f.write_number('Limit load factor',nlim,' ')
    propulsion = data_dict['propulsion'][0]
    f.write_txt('Propulsion system',propulsion)
    var = Variable(data_dict['P0_eng'])
    if propulsion == 'propeller':
        f.write_number('Max engine power',var.value,var.unit,scientific=1)
    else:
        f.write_number('Max engine thrust',var.value,var.unit,scientific=1)
    var.convert2SI()
    P0_eng = var.value
    sfc = float(data_dict['sfc'][0])
    f.write_number('Specific fuel consup',sfc,' ')
    M = float(data_dict['M'][0])
    f.write_number('Max Mach (plots)',M,' ')
    var = Variable(data_dict['h'])
    f.write_number('Max altitude (plots)',var.value,var.unit,scientific=1)
    var.convert2SI()
    h = var.value
    var = Variable(data_dict['h_ref'])
    f.write_number('Ref altitude (plots)',var.value,var.unit,scientific=1)
    var.convert2SI()
    h_ref = var.value
    var = Variable(data_dict['V'])
    f.write_number('Max velocity (plots)',var.value,var.unit)
    var.convert2SI()
    V = var.value
    var = Variable(data_dict['Vcruise'])
    f.write_number('Cruise speed',var.value,var.unit)
    var.convert2SI()
    Vcruise = var.value
    var = Variable(data_dict['VA'])
    f.write_number('Maneuvre speed',var.value,var.unit)
    var.convert2SI()
    VA = var.value
    cd0 = []
    for i,cd in enumerate(data_dict['cd0']):
        cd0.append(float(cd))
        f.write_number('Cd0 condition {}'.format(i+1),cd,' ')
    var = Variable(data_dict['fuel_W'])
    f.write_number('Fuel Weight',var.value,var.unit)
    var.convert2SI()
    fuel_W = var.value
    eta_p = float(data_dict['eta_p'][0])
    f.write_number('Propeller efficiency',eta_p,' ')
    supercharger = int(data_dict['supercharger'][0])
    f.write_number('Supercharger',supercharger,' ')
    plane = Airplane(ac_name,W,S,df,b,c,sweep,theta,numpy.array(clmax),nlim,P0_eng,sfc,propulsion,M,h,h_ref,V,VA,Vcruise,numpy.array(cd0),fuel_W,output,f,eta_p,supercharger)
    plane.steady_level_flight(out_path)
    plane.unpowered_flight(out_path)
    #need to fix the next and the engine power methods
    plane.climb_performance(out_path)
    plane.range_endurance(out_path)
    plane.turn_performance(out_path)
    plane.take_off_landing(out_path)
    return

########################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to plot and extract data related to aircraft performance.')
    parser.add_argument('-dat', nargs='+', default=['/home/jh/Documents/scripts/projects/performance/data_performance.dat'], help='Path to the dat file containing the input variables.')
    parser.add_argument('-o', nargs='+', default=['.'], help='Desired path to save the results.')
    args = parser.parse_args()

    performance_calculations(args.dat[0],args.o[0]) 
