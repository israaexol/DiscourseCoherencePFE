import './App.css';
import {Accueil, Apropos} from './Component';
import { Router, Switch, Route, Redirect, BrowserRouter } from "react-router-dom";

const App = () => {
  return (
    <div className="App">
      <Router>
        <Switch>
          {
            <>
              <Route path='/accueil'>
                <Accueil/>
              </Route>

              <Route path='/apropos'>
                <Apropos/>
              </Route>
            </>
          }
        </Switch>
      </Router>
    </div>
  );
}

export default App;
