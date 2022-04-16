import { useState } from 'react';
import * as  React from 'react';
import axios from 'axios'
import { Form, Row, Col } from "react-bootstrap";
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ClickAwayListener from '@mui/material/ClickAwayListener';
import Grow from '@mui/material/Grow';
import Paper from '@mui/material/Paper';
import Popper from '@mui/material/Popper';
import MenuItem from '@mui/material/MenuItem';
import MenuList from '@mui/material/MenuList';
import './Accueil.css'
import Sidebar from '../Sidebar/Sidebar'
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Brightness1RoundedIcon from '@mui/icons-material/Brightness1Rounded';
import CircularProgress from '@mui/material/CircularProgress';

const Accueil = () => {
  const [text, setText] = useState("");
  const options = ['Parenté sémantique entre les phrases', 'Parenté sémantique entre les paragraphes', 'Parenté sémantique entre les phrases et les paragraphes', 'Richesse lexicale', 'Richesse lexicale et parenté sémantique'];
  const [open, setOpen] = React.useState(false);
  const anchorRef = React.useRef(null);
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const [isLoading, setLoading] = useState(null)
  const handleSubmit = (event) => {
    setLoading(true)
    event.preventDefault();
    // alert(` you entered : ${text}, ${selectedIndex}`);
    const params = { text, selectedIndex };
    var divelement = document.getElementById('evalSection')
    axios
      .post('http://localhost:8080/evaluate/', params)
      .then((res) => {
        const data = res.data.data
        const msg = `${data.score}`
        divelement.hidden = false
        setScore(msg)
        setLoading(false)
      })
      .catch((error) => {
        // alert(`Error: ${error.message}`)
        divelement.hidden = false
        setScore(error.message)
      })

  }

  const handleClick = () => {
  };

  const handleMenuItemClick = (event, index) => {
    setSelectedIndex(index);
    setOpen(false);
  };

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }

    setOpen(false);
  };
  const [scoreResult, setScore] = useState();
  function BasicCard({ score }) {
    let val
    if (score >= 2) {
      val = <Typography variant="h6" component="div" color="#079615">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score >= 1) {
      val = <Typography variant="h6" component="div" color="#FF9A02">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score >= 0) {
      val = <Typography variant="h6" component="div" color="#E33A3A">
        Score de cohérence : {score}
      </Typography>
    }
    else {
      val = <Typography variant="h6" component="div">
        Score de coherence : {score}
      </Typography>
    }
    return (
      <div className='result'>
        <Card sx={{ minWidth: 275, border: 1 }}>
          <CardContent>
            <Typography variant="h6" component="div">
              {val}
            </Typography>
          </CardContent>
          <CardActions sx={{ position: 'relative' }}>
            <Button size="small">Voir plus</Button>
          </CardActions>
        </Card>
      </div>
    );
  }
  function Score({ scoreResult }) {
    switch (scoreResult) {
      case null:
        return <BasicCard score="" />
      default:
        return <BasicCard score={scoreResult} />
    }
  }

  return (
    <>
      <Sidebar selectedIndex={selectedIndex}/>
      <div className='form'>
        <Form onSubmit={handleSubmit}>
          <div className='input_text'>
            <textarea
              id='CheckIt'
              className='_textarea'
              required
              type='text'
              placeholder="Insérez votre texte" 
              value={text}
              onChange={(e) => setText(e.target.value)
              }
            />
          </div>
          <br />
          <Button type="button" id='import_btn'>Importer un fichier</Button>

          <div className='eval_anal'>
            <div id='analyser_btn'>
              <ButtonGroup variant="contained" ref={anchorRef} aria-label="split button">
                <Button onClick={handleClick} resize="none" >{options[selectedIndex]}</Button>
                <Button
                  size="small"
                  aria-controls={open ? 'split-button-menu' : undefined}
                  aria-expanded={open ? 'true' : undefined}
                  aria-label="select merge strategy"
                  aria-haspopup="menu"
                  onClick={handleToggle}
                >
                  <ArrowDropDownIcon />
                </Button>
              </ButtonGroup>
              <Popper
                open={open}
                anchorEl={anchorRef.current}
                role={undefined}
                transition
                disablePortal
              >
                {({ TransitionProps, placement }) => (
                  <Grow
                    {...TransitionProps}
                    style={{
                      transformOrigin:
                        placement === 'bottom' ? 'center top' : 'center bottom',
                    }}
                  >
                    <Paper>
                      <ClickAwayListener onClickAway={handleClose}>
                        <MenuList id="split-button-menu" autoFocusItem>
                          {options.map((option, index) => (
                            <MenuItem
                              key={option}
                              selected={index === selectedIndex}
                              onClick={(event) => handleMenuItemClick(event, index)}
                            >
                              {option}
                            </MenuItem>
                          ))}
                        </MenuList>
                      </ClickAwayListener>
                    </Paper>
                  </Grow>
                )}
              </Popper>
            </div>
            <Button type="submit" id='eval_btn'>Évaluer</Button>
          </div>
          { isLoading == true ? <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '4%' }}><CircularProgress /></Box> :
          <div id='evalSection'>
            <Score scoreResult={scoreResult} />
            <div className='cards-key'>
              <div className='card1'>
                <div><Brightness1RoundedIcon sx={{ color: "#079615" }} /></div>
                <p id='scoreCard'>2 - 3 (élevé)</p>
              </div>
              <div className='card1'>
                <div><Brightness1RoundedIcon sx={{ color: "#FF9A02" }} /></div>
                <p id='scoreCard'>1 - 2 (moyen)</p>
              </div>
              <div className='card1'>
                <div><Brightness1RoundedIcon sx={{ color: "#E33A3A" }} /></div>
                <p id='scoreCard'>0 - 1 (bas)</p>
              </div>
            </div>
          </div> }
        </Form>
      </div>
    </>

  )
}

export default Accueil