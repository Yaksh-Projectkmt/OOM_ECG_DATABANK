class ECGDBRouter:
    route_app_labels = {'analysis_tool', 'morphology_drow', 'oom_ecg_data', 'report'}  # Keep subscription out

    def db_for_read(self, model, **hints):
        if model._meta.app_label in self.route_app_labels:
            return 'mongodb'
        return 'default'

    def db_for_write(self, model, **hints):
        if model._meta.app_label in self.route_app_labels:
            return 'mongodb'
        return 'default'

    def allow_relation(self, obj1, obj2, **hints):
        dbs = {obj1._state.db, obj2._state.db}
        return not ('default' in dbs and 'mongodb' in dbs)

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label in self.route_app_labels:
            return db == 'mongodb'
        return db == 'default'
