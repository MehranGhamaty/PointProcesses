


from PointProcesses.PointProcess import PointProcess 
from DataStores.Trajectory import Trajectory, TimeSlice, Field

def transform_to_uniform(traj: Trajectory, pointprocess: PointProcess):
    """
        Converts a trajectory such that it is uniform, assuming
        that the original distribution of the interarrival times are described 
        properly via the trajectory.
    
        That is I need to be able to integrate the intensity over specific portions of time....

        This makes several other analysis tools easier... I wish I included the papers for this.
    """
