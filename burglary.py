from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def buildBN():
    burglary_model = BayesianModel([('Burglary', 'Alarm'),
                                    ('Earthquake', 'Alarm'),
                                    ("Alarm", "JohnCalls"),
                                    ("Alarm", "MaryCalls")])

    cpd_burg = TabularCPD(variable='Burglary', variable_card=2,
                          values=[[.999], [.001]])                      # [ P(!B), p(B) ]

    cpd_earth = TabularCPD(variable='Earthquake', variable_card=2,
                           values=[[.998], [.002]])                     # [ P(!E), p(E) ]

    cpd_alarm = TabularCPD(variable='Alarm', variable_card=2,
                           values=[[.999, .06, .71, .05],               # P(!A|!E,!B), P(!A|!E,B), P(!A|E,!B), P(!A|E,B)
                                   [.001, .94, .29, .95]],              # P(A|!E,!B), P(A|!E,B), P(A|E,!B), P(A|E,B)
                           evidence=['Earthquake', 'Burglary'],
                           evidence_card=[2, 2])

    cpd_john = TabularCPD(variable="JohnCalls", variable_card=2,
                          values=[[.95, .10], [.05, .90]],              # P(!J|!A), P(!J|A)
                          evidence=['Alarm'], evidence_card=[2])        # P(J|!A), P(J|A)

    cpd_mary = TabularCPD(variable="MaryCalls", variable_card=2,
                          values=[[.99, .30], [.01, .70]],              # P(!M|!A), P(!M|A)
                          evidence=['Alarm'], evidence_card=[2])        # P(M|!A), P(M|A)

    burglary_model.add_cpds(cpd_burg, cpd_earth, cpd_alarm, cpd_john, cpd_mary)

    # print(burglary_model.check_model())
    # print(burglary_model.get_independencies())
    # print(burglary_model.edges())
    # print(burglary_model.get_cpds())



    # Doing exact inference using Variable Elimination
    burglary_infer = VariableElimination(burglary_model)

    # using D-interference to determine conditional dependence of B and E given A is observed
    # print(burglary_model.is_active_trail('Burglary', 'Earthquake'))
    # print(burglary_model.is_active_trail('Burglary', 'Earthquake', observed=['Alarm']))

    # print(burglary_infer.query(variables=['JohnCalls'], joint=False, evidence={'Earthquake': 0})['JohnCalls'])
    # print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'Burglary': 1, 'Earthquake': 0})['MaryCalls'])
    # print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'Burglary': 1, 'Earthquake': 1})['MaryCalls'])
    # print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'JohnCalls': 1})['MaryCalls'])
    # print(burglary_infer.query(variables=['MaryCalls'], joint=False, evidence={'JohnCalls': 1, 'Burglary': 0,"Earthquake": 0})['MaryCalls'])


    return burglary_infer

buildBN()