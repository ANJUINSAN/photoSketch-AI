import React from "react";
import { Link } from "react-router-dom";

import './hero.css';
import hand_icon from "../images/Assets/hand_icon.png";
import arrow_icon from "../images/Assets/arrow.png";
import hero_image from "../images/Assets/hero_images.png";

export default function Hero() {
  return (
    <div className="hero">
      <div className="hero-left">
        <h2>MENS ARRIVALS ONLY</h2>
        <div>
          <div className="hero-hand-icon">
            <p>New</p>
            <img src={hand_icon}></img>
          </div>
          <p>Collections</p>
          <p>for everyone</p>
          <div>
             <Link to={'/mens'} style={{textDecoration:'none'}} > 
            <div className="hero-latest-btn">
             <div>latest collections</div>
              <img src={arrow_icon}></img>
            </div>
             </Link>
          </div>
        </div>
      </div>
      <div className="hero-right">
        <img src={hero_image} alt="" />
      </div>
    </div>
  );
}
