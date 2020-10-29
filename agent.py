from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference


def buildDBN():
    # Construct a DBN object
    dbn = DBN()
    dbn.add_edges_from([(('L', 0), ('O', 0)),
                        (('L', 0), ('L', 1)),
                        (('L', 1), ('O', 1))])


    # setup conditional probability tables for nodes in network

    O0_cpd = TabularCPD(('O', 0), 4, [[0.7, 0.1, 0.1, 0.1],  # A
                                      [0.1, 0.7, 0.1, 0.1],  # B
                                      [0.1, 0.1, 0.7, 0.1],  # C
                                      [0.1, 0.1, 0.1, 0.7]],  # D
                        evidence=[('L', 0)], evidence_card=[4])

    l0_cpd = TabularCPD(('L', 0), 4, [[0],  # A
                                      [0],  # B
                                      [1],  # C
                                      [0]]) # D

    l1_cpd = TabularCPD(('L', 1), 4, [[0.5, 0.0, 0.5, 0.0],  # A
                                      [0.5, 0.5, 0.0, 0.0],  # B
                                      [0.0, 0.0, 0.5, 0.5],  # C
                                      [0.0, 0.5, 0.0, 0.5]], # D
                        evidence=[('L', 0)], evidence_card=[4])

    #add these conditional probability tables to our BayesianModel
    dbn.add_cpds(l0_cpd, l1_cpd, O0_cpd)

    #initialize our model for time series analysis
    dbn.initialize_initial_state()

    # Create an inference object to perform queries
    dbn_inf = DBNInference(dbn)


    #print(dbn_inf.query(variables=[('L',1)], evidence={('O',1): 2})[('L',1)])


    return dbn_inf


buildDBN()
