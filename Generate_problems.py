import optparse
import sys
import random
import os
from Domains.Problem_instance import ProblemInstance as prob

outdir = "Problem_instances/"

# register arguments and set default values
def readCommand(argv):
    parser = optparse.OptionParser(description = 'Create problem instances and store them to file in dir'
                                                 '"Problem_instances".')
    parser.add_option("-o", "--outfile", dest="outfile", type="string",
                      help="write output to FILE", metavar="FILE")
    parser.add_option("-s", "--seed", type="int", dest="seed", default=random.randint(0, 9999999999),
                      help = 'seed integer for random stream')
    parser.add_option("-d", "--domain", dest="domain", type="string",
                      help='problem domain from ' + str(prob.domains))
    parser.add_option("-n", "--size", type="int", dest="size",
                      help='problem size. E.g., number of pancakes in pancake domain or '
                           'dimensionality in tile and rubik')
    parser.add_option("-i", "--instances", type="int", dest="instances", default=1,
                      help='number of problem instances to be created')
    (options, args) = parser.parse_args(argv)
    return options


if __name__ == '__main__':
    options = readCommand(sys.argv)
    assert options.outfile and options.domain and options.size, "arguments must include: outfile, domain, " \
                                                                "and problem size"
    random.seed(options.seed)
    cls = prob.get_domain_class(options.domain)
    with open(os.path.join(outdir, options.outfile + '.txt'), 'w+') as out:
        for i in range(options.instances):
            p = prob()
            p.generate_random(i, cls, cls.get_goal_dummy(options.size))
            out.write(str(p)+'\n')
            print(p)
    print('{} instance of "{}" were created and stored at {}.txt'.format(options.instances,options.domain,
                                                                             options.outfile,))
