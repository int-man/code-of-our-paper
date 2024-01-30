from collections import defaultdict


class Alarm:
    def __init__(self):
        self.id = None
        self.name = None
        self.content = None
        self.kpi_paramters = set()
        self.uid_parameters = set()
        self.template = None
        self.group = set()
        self.template_words = set()
        self.common_parameters = set()
        self.common_semantic = set()
        self.satisfied_pattern = set()
        self.series_group = set()
        self.parameter_group = set()
        self.semantic_group = set()
        self.pattern_group = set()
        self.series = None
        self.incident_id = None
        self.line_id = None
        self.start_time = None
        self.addition = defaultdict(str)

    def __str__(self):
        return 'id=%d,name=%s,parameters=\'%s\',start_time=%s,line_id=%d' % \
               (self.id, self.name, str(self.uid_parameters),
                self.start_time.strftime("%Y/%m/%d %H:%M:%S"), self.line_id)

    def list(self):
        return (self.id, self.name, self.template,
                self.start_time.strftime("%Y/%m/%d %H:%M:%S"),
                self.uid_parameters, self.group, self.auto_group)

    def key_info(self):
        return (self.id, self.name, self.template,
                self.start_time.strftime("%Y/%m/%d %H:%M:%S"), self.uid_parameters)

    def __cmp__(self, other):
        if self.start_time > other.start_time:
            return -1
        elif self.start_time < other.start_time:
            return 1
        elif self.id > other.id:
            return -1
        elif self.id < other.id:
            return 1
        elif self.template > other.template:
            return -1
        elif self.template < other.template:
            return 1
        elif str(self.uid_parameters) > str(other.parameters):
            return -1
        elif str(self.uid_parameters) < str(other.parameters):
            return 1
        else:
            return 0

    def __cmp_time__(self,other):
        if self.start_time != other.start_time:
            return self.start_time < other.start_time
        if self.id != other.id:
            return self.id < other.id
        if self.template != other.template:
            return self.template < other.template
        str_self_parameters = str(self.uid_parameters)
        str_other_parameters = str(other.uid_parameters)
        if str_self_parameters != str_other_parameters:
            return str_self_parameters < str_other_parameters

    def __lt__(self, other):
        if self.start_time != other.start_time:
            return self.start_time < other.start_time
        if self.id != other.id:
            return self.id < other.id
        if self.template != other.template:
            return self.template < other.template
        str_self_parameters = str(self.uid_parameters)
        str_other_parameters = str(other.uid_parameters)
        if str_self_parameters != str_other_parameters:
            return str_self_parameters < str_other_parameters
